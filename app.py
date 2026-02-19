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
DAILY_SOFR = 0.0525 / 252 

# Initialize Loader with Secrets
try:
    fred_key = st.secrets["FRED_API_KEY"]
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Missing Secrets! Please configure FRED_API_KEY and HF_TOKEN in Streamlit.")
    st.stop()

loader = FeatureLoader(fred_key=fred_key, hf_token=hf_token)

# ---------------------------------------------------------------------------
# 2. STRATEGY METHODOLOGY EXPANDER
# ---------------------------------------------------------------------------
st.title("🦅 Multi-Asset SVR + PPO Strategy")

with st.expander("📖 View Strategy & Methodology Details", expanded=False):
    st.markdown("""
    ### 1. Signal Generation: Multi-Target SVR
    We utilize **Support Vector Regression (SVR)** with a Non-Linear RBF Kernel. 
    Models are trained on denoised prices and Macro signals (VIX, DXY, Spreads).
    
    ### 2. Decision Engines
    * **Option A (Pure Momentum):** Allocates to the ticker with the highest positive prediction.
    * **Option B (PPO Overlay):** Mimics a **Proximal Policy Optimization** agent. 
    It requires the SVR signal to exceed a **15bps threshold** to filter market noise and reduce turnover.
    """)

# ---------------------------------------------------------------------------
# 3. CORE BACKTEST ENGINE
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def run_backtest(start_yr, model_choice, t_costs_bps):
    # Load data using the helper function we rectified
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
            st.warning(f"Engine error on {ticker}: {e}")

    pred_df = pd.DataFrame(all_preds).dropna()
    oos_idx = pred_df.index
    
    equity = 100.0
    current_asset = "CASH"
    strat_rets, asset_picks, real_view = [], [], []

    for i in range(len(oos_idx)):
        date = oos_idx[i]
        daily_preds = pred_df.iloc[i]
        
        # Logic Selection
        if "Option B" in model_choice:
            best_ticker = daily_preds.idxmax()
            best_val = daily_preds.max()
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
            day_ret = raw_df.loc[date, ret_col]
        
        equity *= (1 + day_ret)
        strat_rets.append(day_ret)
        asset_picks.append(current_asset)
        real_view.append(day_ret)

    res = pd.DataFrame(index=oos_idx)
    res["Strategy_Path"] = (pd.Series(strat_rets).add(1).cumprod() * 100).values
    res["Allocated_Asset"] = asset_picks
    res["SVR_Predicted"] = pred_df.max(axis=1).values
    res["Realised_Return"] = real_view
    
    # Benchmarks
    for b in ["SPY", "AGG"]:
        res[f"{b}_Benchmark"] = (raw_df.loc[oos_idx, b].pct_change().add(1).cumprod() * 100).values
    
    return res, raw_df

# ---------------------------------------------------------------------------
# 4. SIDEBAR & REFRESH LOGIC
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Admin Controls")
    
    # 1. Data Refresh Button (Restored Logic)
    if st.button("🔄 Data Refresh", use_container_width=True):
        with st.spinner("Syncing incremental data..."):
            status_msg = loader.sync_data()
            if "Success" in status_msg:
                st.success(status_msg)
                st.cache_data.clear()
            elif "Already Up to Date" in status_msg:
                st.info(status_msg)
            else:
                st.error(status_msg)

    st.divider()
    st.header("Settings")
    model_option = st.radio("Model Selection", ["Option A (Pure SVR)", "Option B (SVR + PPO)"])
    start_year = st.slider("Start Year", 2015, 2025, 2020)
    t_costs = st.number_input("T-Costs (bps)", 0, 50, 5)

# ---------------------------------------------------------------------------
# 5. EXECUTION & DISPLAY
# ---------------------------------------------------------------------------
data, raw_full = run_backtest(start_year, model_option, t_costs)

# Last Sync Display
if not raw_full.empty:
    last_date = raw_full.index.max().strftime('%Y-%m-%d')
    st.sidebar.caption(f"Last data point: {last_date}")

# Metrics
m1, m2, m3, m4 = st.columns(4)
ann_ret = (data["Strategy_Path"].iloc[-1] / 100) ** (252 / len(data)) - 1
hit_ratio = (data.tail(15)["SVR_Predicted"].gt(0) == data.tail(15)["Realised_Return"].gt(0)).mean()

m1.metric("Ann. Return", f"{ann_ret:.2%}")
m2.metric("Hit Ratio (15d)", f"{hit_ratio:.0%}")
m3.metric("Final Equity", f"${data['Strategy_Path'].iloc[-1]:.2f}")
m4.metric("Current Pick", data["Allocated_Asset"].iloc[-1])

# Plotly Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Strategy_Path"], name="Strategy", line=dict(width=3, color="#00d4ff")))
fig.add_trace(go.Scatter(x=data.index, y=data["SPY_Benchmark"], name="SPY", line=dict(dash='dot', color='rgba(255,255,255,0.3)')))
fig.add_trace(go.Scatter(x=data.index, y=data["AGG_Benchmark"], name="AGG", line=dict(dash='dot', color='rgba(255,165,0,0.3)')))
fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# Audit Log
st.subheader("📋 15-Day Strategy Audit Log")
audit = data.tail(15).copy()
audit = audit[["Allocated_Asset", "SVR_Predicted", "Realised_Return"]]
st.table(audit.style.format({"SVR_Predicted": "{:.2%}", "Realised_Return": "{:.2%}"}))
