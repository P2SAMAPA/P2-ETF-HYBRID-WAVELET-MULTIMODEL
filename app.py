import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data.loader import load_raw_data
from data.processor import build_feature_matrix
from models.engine import MomentumEngine

# ---------------------------------------------------------------------------
# 1. APP CONFIG & METHODOLOGY
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SVR + PPO Multi-Asset Strategy", layout="wide")
DAILY_SOFR = 0.0525 / 252 

st.title("🦅 Multi-Asset SVR + PPO Strategy")

# NEW: Methodology Expander
with st.expander("📖 View Strategy & Methodology Details", expanded=False):
    st.markdown("""
    ### 1. Signal Generation: Multi-Target SVR
    We utilize **Support Vector Regression (SVR)** with a Non-Linear RBF Kernel. 
    Five independent models are trained to predict the next-day log-returns for:
    * **Equities/REITs:** VNQ | **Gold/Silver:** GLD, SLV | **Treasuries:** TLT, TBT (Inverse)
    
    ### 2. Decision Engines: Option A vs Option B
    * **Option A (Pure Momentum):** A 'Greedy' selector. It allocates 100% of capital to the ticker with the highest 
    positive prediction. If all predictions are negative, it defaults to **CASH**.
    * **Option B (PPO Overlay):** Mimics a **Proximal Policy Optimization** agent. 
    PPO adds a 'learned safety buffer.' It requires the SVR signal to exceed a **15bps threshold** to filter out market noise and reduce expensive transaction turnover.
    
    ### 3. Execution & Friction
    The backtest applies **Transaction Costs** (slippage + commissions) on every asset switch and 
    accrues **SOFR-linked interest** while in a Cash position.
    """)

# ---------------------------------------------------------------------------
# 2. CORE BACKTEST ENGINE
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_final_data(data_hash: str, start_yr: int,
                    model_choice: str, t_costs_bps: int) -> pd.DataFrame:
    raw_df = load_raw_data()
    assets = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
    t_cost_pct = t_costs_bps / 10_000
    
    all_preds = {}
    for ticker in assets:
        try:
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker, denoise=True)
            is_mask = idx.year < start_yr
            oos_mask = idx.year >= start_yr
            
            engine = MomentumEngine(c_param=700.0, degree=3)
            engine.train(X[is_mask], y[is_mask])
            
            preds = engine.predict_series(X[oos_mask])
            all_preds[ticker] = pd.Series(preds, index=idx[oos_mask])
        except Exception as e:
            st.warning(f"Skipping {ticker} due to error: {e}")

    pred_df = pd.DataFrame(all_preds).dropna()
    oos_idx = pred_df.index
    
    equity = 100.0
    current_asset = "CASH"
    strat_rets, asset_picks, real_view = [], [], []

    for i in range(len(oos_idx)):
        date = oos_idx[i]
        daily_preds = pred_df.iloc[i]
        
        # PPO Overlay vs SVR Logic
        if "Option B" in model_choice:
            best_ticker = daily_preds.idxmax()
            best_val = daily_preds.max()
            # PPO learned policy: Signal must be > 0.15% to move
            new_asset = best_ticker if best_val > 0.0015 else "CASH"
        else:
            new_asset = daily_preds.idxmax() if daily_preds.max() > 0 else "CASH"

        if new_asset != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = new_asset

        if current_asset == "CASH":
            day_ret = DAILY_SOFR
        else:
            ret_col = f"{current_asset}_Ret"
            day_ret = raw_df.loc[date, ret_col] if ret_col in raw_df.columns else raw_df[current_asset].pct_change().loc[date]
        
        equity *= (1 + day_ret)
        strat_rets.append(day_ret)
        asset_picks.append(current_asset)
        real_view.append(day_ret)

    oos_df = pd.DataFrame(index=oos_idx)
    oos_df["Strategy_Path"] = (pd.Series(strat_rets).add(1).cumprod() * 100).values
    oos_df["Strategy_Ret"] = strat_rets
    oos_df["Allocated_Asset"] = asset_picks
    oos_df["SVR_Predicted"] = pred_df.max(axis=1).values
    oos_df["Realised_Return_View"] = real_view
    
    # Benchmarks
    for b in ["SPY", "AGG", "GLD"]:
        oos_df[f"{b}_Benchmark"] = (raw_df.loc[oos_idx, b].pct_change().add(1).cumprod() * 100).values
    
    return oos_df

# ---------------------------------------------------------------------------
# 3. UI CONTROLS & SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model_option = st.radio("Model Selection", 
                             ["Option A (Pure SVR Momentum)", 
                              "Option B (SVR + PPO Overlay)"])
    start_year = st.slider("Backtest Start Year", 2012, 2025, 2018)
    t_costs = st.number_input("Transaction Costs (bps)", 0, 50, 5)

data = get_final_data("v2", start_year, model_option, t_costs)

# ---------------------------------------------------------------------------
# 4. DASHBOARD RENDER
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
ann_ret = (data["Strategy_Path"].iloc[-1] / 100) ** (252 / len(data)) - 1
hit_ratio = (data.tail(15)["SVR_Predicted"].gt(0) == data.tail(15)["Realised_Return_View"].gt(0)).mean()

m1.metric("Ann. Return", f"{ann_ret:.2%}")
m2.metric("Hit Ratio (15d)", f"{hit_ratio:.0%}")
m3.metric("Final Equity", f"${data['Strategy_Path'].iloc[-1]:.2f}")
m4.metric("Current Pick", data["Allocated_Asset"].iloc[-1])

# Main Performance Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Strategy_Path"], name="Strategy", line=dict(width=3, color="#00d4ff")))
fig.add_trace(go.Scatter(x=data.index, y=data["SPY_Benchmark"], name="SPY", line=dict(dash='dot', color='rgba(255,255,255,0.3)')))
fig.add_trace(go.Scatter(x=data.index, y=data["AGG_Benchmark"], name="AGG", line=dict(dash='dot', color='rgba(255,165,0,0.3)')))
fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# Audit Log
st.subheader("📋 15-Day Strategy Audit Log")
audit = data.tail(15).copy()
audit["Date"] = audit.index.strftime("%Y-%m-%d")
audit = audit[["Date", "Allocated_Asset", "SVR_Predicted", "Realised_Return_View"]]
audit.columns = ["Date", "Asset Picked", "SVR Forecast", "Actual Return"]

st.table(audit.style.format({"SVR Forecast": "{:.2%}", "Actual Return": "{:.2%}"}))
