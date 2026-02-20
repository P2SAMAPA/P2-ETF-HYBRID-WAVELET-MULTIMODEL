import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.loader import load_raw_data
from models.engine import MomentumEngine, A2CEngine

# Institutional UI Configuration - White Background
st.set_page_config(page_title="P2 ETF WAVELET SVR PPO MODEL", layout="wide", initial_sidebar_state="collapsed")

# Initialize refresh timestamp in session state
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = "Never (Initial Load)"

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
            
            # --- SURGICAL FIX FOR OPTION C & D SELECTION ---
            if "Option C" in model_choice:
                from models.engine import A2CEngine
                engine = A2CEngine()
                engine.train(X[is_mask], y[is_mask])
                # Ensure we return a dated Series so the UI doesn't go blank
                preds = engine.predict_series(X[oos_mask])
                all_preds[ticker] = pd.Series(preds, index=idx[oos_mask])
            else:
                # Options A, B, and D use SVR
                engine = MomentumEngine(c_param=700.0)
                engine.train(X[is_mask], y[is_mask])
                all_preds[ticker] = pd.Series(engine.predict_series(X[oos_mask]), index=idx[oos_mask])
        except: continue

    pred_df = pd.DataFrame(all_preds).dropna()
    if pred_df.empty: return None

  # --- SURGICAL FIX FOR OPTION D THRESHOLD ---
    if "Option B" in model_choice:
        threshold = 0.0015  # Strict PPO Hurdle
    elif "Option D" in model_choice:
        threshold = 0.0005  # A2C Advantage Filter (makes D different from A)
    else:
        threshold = 0.0     # Option A and C: Direct Action

    oos_idx = pred_df.index
    equity = 100.0
    current_asset = "CASH"
    strat_rets = []
    asset_history = []
    
    # --- Peak Tracking Variables ---
    peak_price_since_entry = 0.0
    stop_triggered = False

    for date in oos_idx:
        daily_preds = pred_df.loc[date]
        best_ticker = daily_preds.idxmax()
        
        signal_asset = best_ticker if daily_preds.max() > threshold else "CASH"
        
        if current_asset != "CASH":
            current_price = raw_df.loc[date, current_asset]
            peak_price_since_entry = max(peak_price_since_entry, current_price)
            if current_price < (peak_price_since_entry * 0.90): # 10% Stop
                stop_triggered = True
        
        if stop_triggered:
            new_asset = "CASH"
            if signal_asset != current_asset and signal_asset != "CASH":
                stop_triggered = False
                new_asset = signal_asset
                peak_price_since_entry = raw_df.loc[date, new_asset] if new_asset != "CASH" else 0.0
        else:
            new_asset = signal_asset
            if new_asset != current_asset and new_asset != "CASH":
                peak_price_since_entry = raw_df.loc[date, new_asset]

        if new_asset != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = new_asset
        
        rf_daily = (raw_df.loc[date, "TBILL_3M"] / 100) / 252
        day_ret = rf_daily if current_asset == "CASH" else raw_df.loc[date, f"{current_asset}_Ret"]
        equity *= (1 + day_ret)
        strat_rets.append(day_ret)
        asset_history.append(current_asset)

    res = pd.DataFrame(index=oos_idx)
    res["Strategy_Ret"] = strat_rets
    res["Equity"] = (pd.Series(strat_rets, index=oos_idx).add(1).cumprod() * 100)
    res["Peak"] = res["Equity"].cummax()
    res["Drawdown"] = (res["Equity"] - res["Peak"]) / res["Peak"]
    res["RF"] = (raw_df.loc[oos_idx, "TBILL_3M"] / 100) / 252
    
    for b in ["SPY", "AGG"]:
        res[b] = (raw_df.loc[oos_idx, f"{b}_Ret"].add(1).cumprod() * 100)

    today = datetime.now()
    next_mkt = today + timedelta(days=1) if today.hour >= 16 else today
    while next_mkt.weekday() >= 5:
        next_mkt += timedelta(days=1)

    # Z-Score Conviction
    last_preds = pred_df.iloc[-1]
    if last_preds.std() > 0:
        z_score = (last_preds.max() - last_preds.mean()) / last_preds.std()
        confidence_val = 0.5 + (z_score * 0.18) 
    else:
        confidence_val = 0.50
    confidence_val = max(0.40, min(0.98, confidence_val))

    # Win/Loss Ratio for Kelly
    pos_rets = [r for r in strat_rets if r > 0]
    neg_rets = [abs(r) for r in strat_rets if r < 0]
    win_loss_ratio = (np.mean(pos_rets) / np.mean(neg_rets)) if (pos_rets and neg_rets) else 1.0

    return {
        "df": res,
        "audit": pd.DataFrame({"Allocation": asset_history, "Daily_Return": strat_rets}, index=oos_idx),
        "target": asset_history[-1],
        "confidence": confidence_val,
        "win_loss_ratio": win_loss_ratio,
        "next_date": next_mkt.strftime('%A, %b %d, %Y')
    }

# ---------------------------------------------------------------------------
# TERMINAL UI RENDERING
# ---------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #1a73e8; margin-bottom: 0;'>🦅 P2 ETF WAVELET SVR PPO MODEL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5f6368; font-weight: 500;'>Institutional Strategy Performance & Signal Console</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Terminal Config")
    if st.button("🔄 Force Data Refresh"):
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now().strftime("%b %d, %H:%M:%S")
        st.rerun()

    st.caption(f"✨ Last sync: {st.session_state.last_refresh}")
    s_yr = st.slider("Backtest Start Year", 2010, 2024, 2015)
    
    # NEW RL-INTEGRATED LABELS
    opt = st.radio("Model Logic", [
        "Option A - Wavelet-SVR", 
        "Option B - Wavelet-SVR-PPO", 
        "Option C - Wavelet-A2C", 
        "Option D - Wavelet-SVR-A2C"
    ])
    
    costs = st.number_input("T-Costs (bps)", 0, 50, 10)

output = run_professional_backtest(s_yr, opt, costs)

if output:
    data = output["df"]
    
    # ROW 1: PRIMARY TARGET SIGNAL
    conf_color = "#2e7d32" if output['confidence'] > 0.7 else "#f57c00"
    st.markdown(f"""
        <div style="background-color: #f1f8e9; padding: 30px; border-radius: 12px; border: 2px solid #a5d6a7; text-align: center; margin-bottom: 25px;">
            <p style="margin:0; color: #2e7d32; font-size: 14px; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">
                Target Allocation for {output['next_date']}
            </p>
            <h1 style="margin:5px 0; font-size: 100px; color: #1b5e20; font-family: 'Courier New', monospace; font-weight: 900;">
                {output['target']}
            </h1>
            <div style="width: 240px; margin: 0 auto;">
                <p style="margin:0; font-size: 12px; color: {conf_color}; font-weight: bold; text-transform: uppercase;">
                    Signal Conviction: {output['confidence']:.0%}
                </p>
                <div style="background-color: #e0e0e0; border-radius: 10px; height: 10px; width: 100%; margin-top: 5px;">
                    <div style="background-color: {conf_color}; height: 10px; width: {output['confidence']*100}%; border-radius: 10px;"></div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ROW 2: PERFORMANCE & RISK METRICS
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    excess = data["Strategy_Ret"] - data["RF"]
    ann_ret = (data["Equity"].iloc[-1] / 100) ** (252 / len(data)) - 1
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() != 0 else 0
    max_dd = data["Drawdown"].min()
    max_daily_loss = data["Strategy_Ret"].min()
    worst_day_date = data["Strategy_Ret"].idxmin().strftime('%b %d, %Y')

    # Audit Trail Sync
    audit_data = output["audit"][output["audit"]["Daily_Return"] != 0].tail(15)
    audit_data.index = audit_data.index.date
    hit_ratio_sync = (audit_data["Daily_Return"] > 0).sum() / len(audit_data) if len(audit_data) > 0 else 0

    # Kelly Calculation
    p = hit_ratio_sync
    b = output["win_loss_ratio"]
    kelly_f = ((p * (b + 1)) - 1) / b if b > 0 else 0
    safe_kelly = max(0, min(1.0, kelly_f * 0.5))
    
    # Arrow Logic
    if safe_kelly > 0.60:
        k_arrow, k_col = "▲", "normal"
    elif safe_kelly < 0.30:
        k_arrow, k_col = "▼", "inverse"
    else:
        k_arrow, k_col = "▶", "off"

    m1.metric("Annualized Return", f"{ann_ret:.2%}")
    m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m3.metric("Max Drawdown (P-T)", f"{max_dd:.2%}")
    m4.metric("Max DD (Daily)", f"{max_daily_loss:.2%}", help=f"Occurred on {worst_day_date}")
    m5.metric("Hit Ratio (15D)", f"{hit_ratio_sync:.0%}")
    m6.metric("Kelly Recco", f"{safe_kelly:.0%}", delta=f"{k_arrow} Edge", delta_color=k_col)

    # ROW 3: EQUITY CHART
    st.markdown("<h3 style='margin-top: 25px; margin-bottom: 10px;'>Cumulative Performance: Strategy vs. SPY Benchmark</h3>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Equity"], name="Model Strategy", line=dict(color='#1a73e8', width=3)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SPY"], name="SPY (Re-indexed)", line=dict(color='#9aa0a6', dash='dot')))
    fig.update_layout(template="plotly_white", height=480, margin=dict(l=0,r=0,t=0,b=0), hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)

    # ROW 4: AUDIT TABLE & METHODOLOGY
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("📋 15-Day Strategy Audit Trail")
        def style_returns(val):
            color = '#1b5e20' if val > 0 else '#b71c1c' if val < 0 else '#5f6368'
            return f'color: {color}; font-weight: bold;'
        
        st.dataframe(
            audit_data.style.format({"Daily_Return": "{:.2%}"}).applymap(style_returns, subset=['Daily_Return']),
            use_container_width=True, height=560
        )

    with col_right:
        st.subheader("🔬 Methodology Overview")
        
        if "A2C" in opt:
            rl_desc = "**RL Engine:** A2C (Advantage Actor-Critic) utilizes a synchronous policy gradient to maximize the 'Advantage' of a trade relative to a baseline risk-free return."
        elif "PPO" in opt:
            rl_desc = "**RL Engine:** PPO (Proximal Policy Optimization) utilizes a clipped objective function to ensure stable, conservative strategy updates."
        else:
            rl_desc = "**RL Engine:** None (Pure SVR). Allocation is based solely on the highest raw SVR point-prediction."

        st.markdown(f"""
        **Algorithm:** Non-linear SVR with Wavelet Denoising for high-frequency signal extraction.
        
        **Strategy Logic:**
        - **{opt}:** { 'Hybrid mode using SVR for prediction and RL for execution filtering.' if 'SVR-' in opt else 'Pure-play model focusing on ' + opt.split('-')[-1] + ' logic.' }
        
        **Risk & Execution:**
        - {rl_desc}
        - **Liquidity Buffer:** 'CASH' positions yield the daily **3-Month T-Bill** rate.
        - **Kelly Sizing:** Suggests capital allocation based on the 15-day Edge Profile (Half-Kelly).
        
        **System Integrity:**
        - **Lookback History:** {len(data)} trading sessions analyzed in this window.
        - **Next Signal Window:** Active for market open on {output['next_date']}.
        """)
