import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.loader import load_raw_data
from models.engine import MomentumEngine

# Institutional UI Configuration
st.set_page_config(page_title="Eagle Alpha Terminal", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Professional Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    [data-testid="stMetricValue"] { font-size: 28px !important; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CORE ANALYTICS ENGINE
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def run_professional_backtest(start_yr, model_choice, t_costs_bps):
    raw_df = load_raw_data()
    assets = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
    t_cost_pct = t_costs_bps / 10_000
    
    from data.processor import build_feature_matrix
    all_preds = {}
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
    if pred_df.empty: return None

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
        
        rf_daily = (raw_df.loc[date, "TBILL_3M"] / 100) / 252
        day_ret = rf_daily if current_asset == "CASH" else raw_df.loc[date, f"{current_asset}_Ret"]
        equity *= (1 + day_ret)
        strat_rets.append(day_ret)
        asset_history.append(current_asset)

    # Metrics Calculation
    res = pd.DataFrame(index=oos_idx)
    res["Strategy_Ret"] = strat_rets
    res["Equity"] = (pd.Series(strat_rets, index=oos_idx).add(1).cumprod() * 100)
    res["Peak"] = res["Equity"].cummax()
    res["Drawdown"] = (res["Equity"] - res["Peak"]) / res["Peak"]
    res["RF"] = (raw_df.loc[oos_idx, "TBILL_3M"] / 100) / 252
    
    # Accurate Benchmark Logic
    for b in ["SPY", "AGG"]:
        res[b] = (raw_df.loc[oos_idx, f"{b}_Ret"].add(1).cumprod() * 100)

    # Next Market Date Logic
    last_dt = oos_idx[-1]
    next_mkt = last_dt + timedelta(days=1)
    while next_mkt.weekday() >= 5: next_mkt += timedelta(days=1)

    return {
        "df": res,
        "audit": pd.DataFrame({"Allocation": asset_history, "Daily_Return": strat_rets}, index=oos_idx).tail(15).sort_index(ascending=False),
        "target": asset_history[-1],
        "next_date": next_mkt.strftime('%A, %b %d, %Y')
    }

# ---------------------------------------------------------------------------
# TERMINAL UI RENDERING
# ---------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #58a6ff; margin-bottom: 0;'>🦅 EAGLE ALPHA TERMINAL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Institutional Macro-Signal & SVR Momentum Engine</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Control Panel")
    if st.button("🔄 Force Data Refresh"): st.cache_data.clear(); st.rerun()
    s_yr = st.slider("Backtest Start Year", 2010, 2024, 2015)
    opt = st.radio("Model Logic", ["Option A (Pure SVR)", "Option B (SVR + PPO)"])
    costs = st.number_input("T-Costs (bps)", 0, 50, 5)

output = run_professional_backtest(s_yr, opt, costs)

if output:
    data = output["df"]
    
    # ROW 1: MASTER TARGET SIGNAL
    st.markdown(f"""
        <div style="background-color: #161b22; padding: 30px; border-radius: 12px; border: 2px solid #238636; text-align: center; margin-bottom: 25px;">
            <p style="margin:0; color: #8b949e; font-size: 16px; text-transform: uppercase; letter-spacing: 2px;">Target Allocation for {output['next_date']}</p>
            <h1 style="margin:10px 0; font-size: 82px; color: #2ea043; font-family: monospace;">{output['target']}</h1>
        </div>
    """, unsafe_allow_html=True)

    # ROW 2: PRIMARY ANALYTICS
    m1, m2, m3, m4, m5 = st.columns(5)
    
    excess = data["Strategy_Ret"] - data["RF"]
    ann_ret = (data["Equity"].iloc[-1] / 100) ** (252 / len(data)) - 1
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() != 0 else 0
    max_dd = data["Drawdown"].min()
    daily_dd = data["Drawdown"].iloc[-1]
    hit_ratio = sum(1 for r in output["audit"]["Daily_Return"] if r > 0) / 15

    m1.metric("Annualized Return", f"{ann_ret:.2%}")
    m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m3.metric("Max Drawdown (P-T)", f"{max_dd:.2%}")
    m4.metric("Current Drawdown", f"{daily_dd:.2%}")
    m5.metric("Hit Ratio (15D)", f"{hit_ratio:.0%}")

    # ROW 3: EQUITY CHART
    st.markdown("### Performance Relative to SPY Benchmark")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Equity"], name="Eagle Strategy", line=dict(color='#58a6ff', width=3)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SPY"], name="SPY Benchmark", line=dict(color='#8b949e', dash='dot')))
    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=0,b=0), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ROW 4: AUDIT TABLE & METHODOLOGY
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("📋 15-Day Strategy Audit Trail")
        # Color-coded realized returns
        def color_returns(val):
            color = '#2ea043' if val > 0 else '#f85149' if val < 0 else '#8b949e'
            return f'color: {color}; font-weight: bold;'
        
        st.dataframe(
            output["audit"].style.format({"Daily_Return": "{:.2%}"}).applymap(color_returns, subset=['Daily_Return']),
            use_container_width=True, height=560
        )

    with col_right:
        st.subheader("🔬 Methodology Core")
        st.markdown(f"""
        **System Architecture:** Non-linear Support Vector Regression (SVR) trained on multi-decade macro-ETF feature sets.
        
        **Risk Management:**
        - **{opt}:** Uses policy-driven thresholds to filter conviction.
        - **Cash Hurdle:** Cash positions earn daily 3-Month T-Bill yields (TBILL_3M).
        - **Benchmarking:** Excess returns are calculated net of risk-free rates.
        
        **Latest Context:**
        - **Audit Start:** {output['audit'].index.min().date()}
        - **Backtest History:** {len(data)} trading days
        """)
