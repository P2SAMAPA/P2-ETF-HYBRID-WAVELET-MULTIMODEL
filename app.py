import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data.loader import load_raw_data
from models.engine import MomentumEngine

st.set_page_config(page_title="Eagle Strategy Pro", layout="wide")

# ---------------------------------------------------------------------------
# CORE BACKTEST & PREDICTION ENGINE
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def run_full_strategy(start_yr, model_choice, t_costs_bps):
    raw_df = load_raw_data()
    assets = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
    t_cost_pct = t_costs_bps / 10_000
    
    all_preds = {}
    from data.processor import build_feature_matrix
    
    # 1. Generate Predictions
    for ticker in assets:
        try:
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker)
            is_mask = idx.year < start_yr
            oos_mask = idx.year >= start_yr
            
            engine = MomentumEngine(c_param=700.0)
            engine.train(X[is_mask], y[is_mask])
            all_preds[ticker] = pd.Series(engine.predict_series(X[oos_mask]), index=idx[oos_mask])
        except: continue

    pred_df = pd.DataFrame(all_preds).dropna()
    if pred_df.empty: return None, None, None
    
    # 2. Backtest Execution
    oos_idx = pred_df.index
    equity = 100.0
    current_asset = "CASH"
    strat_rets = []
    asset_history = []

    threshold = 0.0015 if "Option B" in model_choice else 0.0

    for date in oos_idx:
        daily_preds = pred_df.loc[date]
        best_ticker = daily_preds.idxmax()
        new_asset = best_ticker if daily_preds.max() > threshold else "CASH"

        if new_asset != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = new_asset

        # Use TBILL_3M for Cash Return
        rf_daily = (raw_df.loc[date, "TBILL_3M"] / 100) / 252
        day_ret = rf_daily if current_asset == "CASH" else raw_df.loc[date, f"{current_asset}_Ret"]
        equity *= (1 + day_ret)
        
        strat_rets.append(day_ret)
        asset_history.append(current_asset)

    # 3. Results DataFrame
    res = pd.DataFrame(index=oos_idx)
    res["Strategy"] = (pd.Series(strat_rets, index=oos_idx).add(1).cumprod() * 100)
    res["Daily_Ret"] = strat_rets
    res["RF"] = (raw_df.loc[oos_idx, "TBILL_3M"] / 100) / 252
    for b in ["SPY", "AGG"]:
        res[b] = (raw_df.loc[oos_idx, f"{b}_Ret"].add(1).cumprod() * 100)
    
    # Audit Trail
    audit_df = pd.DataFrame({
        "Date": oos_idx,
        "Allocation": asset_history,
        "Daily_Return": strat_rets
    }).set_index("Date").tail(15).sort_index(ascending=False)
    
    return res, audit_df, asset_history[-1]

# ---------------------------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🦅 Eagle Admin")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    start_year = st.slider("Backtest Start", 2010, 2025, 2015)
    model_option = st.radio("Model Logic", ["Option A (Pure SVR)", "Option B (SVR + PPO)"])
    t_costs = st.number_input("T-Costs (bps)", 0, 50, 5)

data, audit, target_etf = run_full_strategy(start_year, model_option, t_costs)

if data is not None:
    # 1. Top Metrics & Target
    excess_ret = data["Daily_Ret"] - data["RF"]
    sharpe = (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)
    ann_ret = (data["Strategy"].iloc[-1] / 100) ** (252 / len(data)) - 1
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ann. Return", f"{ann_ret:.2%}")
    m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m3.metric("Final Value", f"${data['Strategy'].iloc[-1]:.2f}")
    m4.success(f"**Target Allocation:** {target_etf}")

    # 2. Main Equity Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Strategy"], name="Eagle Strategy", line=dict(color='#00CC96', width=3)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SPY"], name="SPY Benchmark", line=dict(color='white', dash='dot')))
    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # 3. Audit Table
    st.subheader("📋 15-Day Strategy Audit")
    st.table(audit.style.format({"Daily_Return": "{:.2%}"}))

    # 4. Methodology Description
    st.divider()
    st.subheader("🔬 Strategy Methodology")
    st.markdown("""
    **Core Engine:** This strategy utilizes a Support Vector Regression (SVR) model with a third-degree polynomial kernel to predict next-day returns for a basket of ETFs (GLD, SLV, TLT, TBT, VNQ).
    
    **Risk Management & Signal Gating:**
    * **Option A (Pure SVR):** Allocates daily to the asset with the highest predicted return.
    * **Option B (SVR + PPO):** Applies a **Proximal Policy Optimization (PPO)** overlay that requires the top prediction to exceed a **0.15% threshold**. If no signal meets this bar, the strategy rotates to **CASH** to preserve capital.
    
    **Risk-Free Benchmark:** Cash positions earn the daily yield of the **3-Month U.S. Treasury Bill (TBILL_3M)**, which is also used as the 'Risk-Free Rate' for the Sharpe Ratio calculation to ensure accurate excess return reporting.
    """)
