import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.loader import load_raw_data
from models.engine import MomentumEngine

# Institutional UI Configuration - White Background
st.set_page_config(page_title="P2 ETF WAVELET SVR PPO MODEL", layout="wide", initial_sidebar_state="collapsed")

# Professional Styling for Metrics and Layout
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    [data-testid="stMetricValue"] { font-size: 32px !important; color: #1a73e8 !important; }
    [data-testid="stMetricLabel"] { color: #5f6368 !important; font-weight: 600 !important; text-transform: uppercase; font-size: 12px; }
    h1, h2, h3 { color: #202124 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stMarkdown p { color: #3c4043 !important; }
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
    
    # --- NEW: Peak Tracking Variables ---
    peak_price_since_entry = 0.0
    stop_triggered = False

    for date in oos_idx:
        daily_preds = pred_df.loc[date]
        best_ticker = daily_preds.idxmax()
        
        # 1. Get the Raw SVR Signal
        signal_asset = best_ticker if daily_preds.max() > threshold else "CASH"
        
        # 2. Check Trailing Stop if we are currently holding an ETF
        if current_asset != "CASH":
            current_price = raw_df.loc[date, current_asset]
            peak_price_since_entry = max(peak_price_since_entry, current_price)
            
            if current_price < (peak_price_since_entry * 0.90): # 10% Stop
                stop_triggered = True
        
        # 3. Decision Logic
        if stop_triggered:
            new_asset = "CASH"
            # Reset stop only if the model points to a DIFFERENT asset
            if signal_asset != current_asset and signal_asset != "CASH":
                stop_triggered = False
                new_asset = signal_asset
                peak_price_since_entry = raw_df.loc[date, new_asset] if new_asset != "CASH" else 0.0
        else:
            new_asset = signal_asset
            if new_asset != current_asset and new_asset != "CASH":
                peak_price_since_entry = raw_df.loc[date, new_asset]

        # 4. Execute Trade (T-Costs)
        if new_asset != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = new_asset
        
        # Cash Return from 3-Month T-Bill
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
    
    for b in ["SPY", "AGG"]:
        res[b] = (raw_df.loc[oos_idx, f"{b}_Ret"].add(1).cumprod() * 100)

    # Date awareness for target signal
    last_dt = oos_idx[-1]
    # Force the date to look forward from the actual current day
    today = datetime.now()
    # If past market close (4 PM), look to tomorrow, otherwise look at today
    next_mkt = today + timedelta(days=1) if today.hour >= 16 else today
    
    # Skip weekends
    while next_mkt.weekday() >= 5:
        next_mkt += timedelta(days=1)

    return {
        "df": res,
        "audit": pd.DataFrame({"Allocation": asset_history, "Daily_Return": strat_rets}, index=oos_idx).tail(15).sort_index(ascending=False),
        "target": asset_history[-1],
        "next_date": next_mkt.strftime('%A, %b %d, %Y')
    }

# ---------------------------------------------------------------------------
# TERMINAL UI RENDERING
# ---------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #1a73e8; margin-bottom: 0;'>🦅 P2 ETF WAVELET SVR PPO MODEL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5f6368; font-weight: 500;'>Institutional Strategy Performance & Signal Console</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Terminal Config")
    if st.button("🔄 Force Data Refresh"): st.cache_data.clear(); st.rerun()
    s_yr = st.slider("Backtest Start Year", 2010, 2024, 2015)
    opt = st.radio("Model Logic", ["Option A (Pure SVR)", "Option B (SVR + PPO)"])
    costs = st.number_input("T-Costs (bps)", 0, 50, 10)

output = run_professional_backtest(s_yr, opt, costs)

if output:
    data = output["df"]
    
    # ROW 1: PRIMARY TARGET SIGNAL
    st.markdown(f"""
        <div style="background-color: #f1f8e9; padding: 35px; border-radius: 12px; border: 2px solid #a5d6a7; text-align: center; margin-bottom: 25px;">
            <p style="margin:0; color: #2e7d32; font-size: 16px; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">Target Allocation for Market Open: {output['next_date']}</p>
            <h1 style="margin:10px 0; font-size: 90px; color: #1b5e20; font-family: 'Courier New', monospace; font-weight: 900;">{output['target']}</h1>
        </div>
    """, unsafe_allow_html=True)

    # ROW 2: PERFORMANCE & RISK METRICS
    m1, m2, m3, m4, m5 = st.columns(5)
    
    excess = data["Strategy_Ret"] - data["RF"]
    ann_ret = (data["Equity"].iloc[-1] / 100) ** (252 / len(data)) - 1
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() != 0 else 0
    max_dd = data["Drawdown"].min()
   # Calculates the single worst 24-hour drop in the history of the backtest
    max_daily_loss = data["Strategy_Ret"].min()
   # Synchronize Hit Ratio directly to the Audit Table display
    # We count positive returns from the exact same dataframe the table uses
   # Filter out rows where returns are exactly zero (holidays/non-trading days)
    audit_data = output["audit"][output["audit"]["Daily_Return"] != 0].tail(15)
    positive_days = (audit_data["Daily_Return"] > 0).sum()
   # Filter out rows where returns are zero (holidays)
    audit_data = output["audit"][output["audit"]["Daily_Return"] != 0].tail(15)
    
    # Remove timestamps from the Date index for a cleaner look
    audit_data.index = audit_data.index.date
    
    # Calculate Hit Ratio from the filtered trading days only
    positive_days = (audit_data["Daily_Return"] > 0).sum()
    hit_ratio_sync = positive_days / len(audit_data) if len(audit_data) > 0 else 0

    # UI Metrics
    m1.metric("Annualized Return", f"{ann_ret:.2%}")
    m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m3.metric("Max Drawdown (P-T)", f"{max_dd:.2%}")
    m4.metric("Max DD (Daily)", f"{max_daily_loss:.2%}")
    m5.metric("Hit Ratio (15D)", f"{hit_ratio_sync:.0%}")
    # ROW 3: EQUITY CHART
    st.markdown("<h3 style='margin-top: 25px; margin-bottom: 10px;'>Cumulative Performance: Strategy vs. SPY Benchmark</h3>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Equity"], name="Eagle Strategy", line=dict(color='#1a73e8', width=3)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SPY"], name="SPY (Re-indexed)", line=dict(color='#9aa0a6', dash='dot')))
    fig.update_layout(
        template="plotly_white", 
        height=480, 
        margin=dict(l=0,r=0,t=0,b=0), 
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ROW 4: AUDIT TABLE & METHODOLOGY
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("📋 15-Day Strategy Audit Trail")
        def style_returns(val):
            color = '#1b5e20' if val > 0 else '#b71c1c' if val < 0 else '#5f6368'
            return f'color: {color}; font-weight: bold;'
        
      
    st.header("📋 15-Day Strategy Audit Trail")
    # Ensure this line starts at the same indentation level as st.header
    st.dataframe(
    audit_data.style.format({"Daily_Return": "{:.2%}"}).applymap(style_returns, subset=['Daily_Return']),
    use_container_width=True, 
    height=560
)

    with col_right:
        st.subheader("🔬 Methodology Overview")
        st.markdown(f"""
        **Algorithm:** Non-linear SVR (Support Vector Regression) utilizing a polynomial kernel to capture non-linear macro relationships.
        
        **Risk Controls:**
        - **Model Selection:** {opt} manages churn based on prediction conviction.
        - **Liquidity Buffer:** 'CASH' positions yield the daily **3-Month T-Bill** rate.
        - **Hurdle Rate:** Sharpe Ratio is calculated net of the Risk-Free Rate (RF).
        
        **System Integrity:**
        - **Feature Space:** Aggregated macro signals and ETF returns.
        - **Lookback History:** {len(data)} trading sessions analyzed in this window.
        - **Next Signal Window:** Active for market open on {output['next_date']}.
        """)
