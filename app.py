import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data.loader import FeatureLoader, load_raw_data
from data.processor import build_feature_matrix
from models.engine import MomentumEngine

# ---------------------------------------------------------------------------
# 1. APP CONFIG & INITIALIZATION
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SVR + PPO Multi-Asset Strategy", layout="wide")

try:
    fred_key = st.secrets["FRED_API_KEY"]
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Missing Secrets! Configure FRED_API_KEY and HF_TOKEN.")
    st.stop()

loader = FeatureLoader(fred_key=fred_key, hf_token=hf_token)

# ---------------------------------------------------------------------------
# 2. CORE BACKTEST ENGINE (Updated for T-Bill Cash)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def run_backtest(start_yr, model_choice, t_costs_bps):
    raw_df = load_raw_data()
    assets = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
    t_cost_pct = t_costs_bps / 10_000
    
    all_preds = {}
    for ticker in assets:
        try:
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker, denoise=True)
            is_mask = idx.year < start_yr
            oos_mask = idx.year >= start_yr
            
            if len(X[is_mask]) < 50: # Safety check for training data
                continue

            engine = MomentumEngine(c_param=700.0, degree=3)
            engine.train(X[is_mask], y[is_mask])
            all_preds[ticker] = pd.Series(engine.predict_series(X[oos_mask]), index=idx[oos_mask])
        except Exception as e:
            st.warning(f"Engine error on {ticker}: {e}")

    pred_df = pd.DataFrame(all_preds).dropna()
    if pred_df.empty: return pd.DataFrame(), raw_df
    
    oos_idx = pred_df.index
    equity = 100.0
    current_asset = "CASH"
    strat_rets = []

    for i in range(len(oos_idx)):
        date = oos_idx[i]
        daily_preds = pred_df.iloc[i]
        
        # PPO Overlay / Signal Gate
        best_ticker = daily_preds.idxmax()
        best_val = daily_preds.max()
        
        if "Option B" in model_choice:
            new_asset = best_ticker if best_val > 0.0015 else "CASH"
        else:
            new_asset = best_ticker if best_val > 0 else "CASH"

        if new_asset != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = new_asset

        # --- DYNAMIC RISK-FREE RATE (T-BILL) ---
        # TBILL_3M is an annualized % (e.g., 4.5). Convert to daily decimal.
        daily_tbill = (raw_df.loc[date, "TBILL_3M"] / 100) / 252 if "TBILL_3M" in raw_df.columns else 0.0001
        
        if current_asset == "CASH":
            day_ret = daily_tbill
        else:
            day_ret = raw_df.loc[date, f"{current_asset}_Ret"]
        
        equity *= (1 + day_ret)
        strat_rets.append(day_ret)

    res = pd.DataFrame(index=oos_idx)
    res["Strategy_Path"] = (pd.Series(strat_rets).add(1).cumprod() * 100).values
    res["Daily_Ret"] = strat_rets
    res["TBILL_Rate"] = (raw_df.loc[oos_idx, "TBILL_3M"] / 100) / 252
    
    for b in ["SPY", "AGG"]:
        res[f"{b}_Benchmark"] = (raw_df.loc[oos_idx, b].pct_change().add(1).cumprod() * 100).values
    
    return res, raw_df

# ---------------------------------------------------------------------------
# 3. SIDEBAR & METRICS (Sharpe Ratio Calculation)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Admin")
    if st.button("🔄 Data Refresh"):
        status = loader.sync_data()
        st.info(status)
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    model_option = st.radio("Model", ["Option A (Pure SVR)", "Option B (SVR + PPO)"])
    start_year = st.slider("Start Year", 2010, 2025, 2018)
    t_costs = st.number_input("T-Costs (bps)", 0, 50, 5)

data, raw_full = run_backtest(start_year, model_option, t_costs)

if not data.empty:
    st.title("🦅 Multi-Asset SVR + PPO Strategy")
    
    # SHARPE RATIO CALCULATION
    # Excess Return = Strategy Return - T-Bill Rate
    excess_ret = data["Daily_Ret"] - data["TBILL_Rate"]
    sharpe = (excess_ret.mean() / excess_ret.std()) * np.sqrt(252) if excess_ret.std() != 0 else 0
    
    ann_ret = (data["Strategy_Path"].iloc[-1] / 100) ** (252 / len(data)) - 1
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Ann. Return", f"{ann_ret:.2%}")
    m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m3.metric("Final Equity", f"${data['Strategy_Path'].iloc[-1]:.2f}")

    st.plotly_chart(go.Figure([
        go.Scatter(x=data.index, y=data["Strategy_Path"], name="Strategy"),
        go.Scatter(x=data.index, y=data["SPY_Benchmark"], name="SPY", line=dict(dash='dot'))
    ]).update_layout(template="plotly_dark"))
else:
    st.warning("Please click 'Data Refresh' to sync T-Bill history and populate the backtest.")
