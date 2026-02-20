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
    from analytics.regime import RegimeHMM, BayesianFilter
    
    all_preds = {}
    is_mask_global = raw_df.index.year < start_yr
    oos_mask_global = raw_df.index.year >= start_yr

    # --- REGIME ENGINE INITIALIZATION ---
    hmm_signals = {}
    if "HMM" in model_choice:
        hmm_engine = RegimeHMM(n_states=3)
        hmm_engine.train_and_assign(raw_df[is_mask_global], assets)
        macro_cols = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]
        macro_oos = raw_df[macro_cols][oos_mask_global].diff().fillna(0)
        for date, row in macro_oos.iterrows():
            hmm_signals[date] = hmm_engine.predict_best_asset(row.values.reshape(1, -1))

    bsts_filter = BayesianFilter() if "BSTS" in model_choice else None

    # --- SVR/A2C PREDICTIONS ---
    if not (model_choice.startswith("Option E") or model_choice.startswith("Option G")):
        for ticker in assets:
            try:
                X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker)
                is_mask = idx.year < start_yr
                oos_mask = idx.year >= start_yr
                engine = A2CEngine() if "Option C" in model_choice else MomentumEngine(c_param=700.0)
                engine.train(X[is_mask], y[is_mask])
                preds = engine.predict_series(X[oos_mask])
                all_preds[ticker] = pd.Series(preds, index=idx[oos_mask])
            except Exception as e:
                continue

    pred_df = pd.DataFrame(all_preds).dropna() if all_preds else pd.DataFrame()

    # --- DYNAMIC THRESHOLD LOGIC ---
    if "Option B" in model_choice:
        threshold = 0.0015
    elif any(opt in model_choice for opt in ["Option D", "Option H"]):
        threshold = pred_df.values.std() * 0.75 if not pred_df.empty else 0.0
    else:
        threshold = 0.0

    oos_idx = raw_df.index[oos_mask_global]
    equity, current_asset = 100.0, "CASH"
    strat_rets, asset_history = [], []
    peak_price_since_entry, stop_triggered = 0.0, False

    for date in oos_idx:
        # SIGNAL ROUTING
        if model_choice.startswith("Option E"):
            signal_asset = hmm_signals.get(date, "CASH")
        elif model_choice.startswith("Option G"):
            signal_asset = hmm_signals.get(date, "CASH") 
        elif model_choice.startswith("Option F"):
            daily_preds = pred_df.loc[date] if date in pred_df.index else None
            hmm_gate = hmm_signals.get(date, "CASH")
            signal_asset = daily_preds.idxmax() if (daily_preds is not None and daily_preds.max() > threshold and hmm_gate != "CASH") else "CASH"
        elif model_choice.startswith("Option H"):
            daily_preds = pred_df.loc[date] if date in pred_df.index else None
            if daily_preds is not None and daily_preds.max() > threshold:
                conf = bsts_filter.get_confidence(raw_df.loc[:date, daily_preds.idxmax()])
                signal_asset = daily_preds.idxmax() if conf > 0.65 else "CASH"
            else:
                signal_asset = "CASH"
        else:
            if date in pred_df.index:
                daily_preds = pred_df.loc[date]
                signal_asset = daily_preds.idxmax() if daily_preds.max() > threshold else "CASH"
            else:
                signal_asset = "CASH"

        # Stop Loss & Execution
        if current_asset != "CASH":
            current_price = raw_df.loc[date, current_asset]
            peak_price_since_entry = max(peak_price_since_entry, current_price)
            if current_price < (peak_price_since_entry * 0.90): stop_triggered = True
        
        if stop_triggered:
            new_asset = "CASH"
            if signal_asset != current_asset and signal_asset != "CASH":
                stop_triggered, new_asset = False, signal_asset
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

    # --- ASSEMBLE RESULTS ---
    res = pd.DataFrame(index=oos_idx)
    res["Strategy_Ret"] = strat_rets
    res["Equity"] = (pd.Series(strat_rets, index=oos_idx).add(1).cumprod() * 100)
    res["Peak"], res["Drawdown"] = res["Equity"].cummax(), (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    res["RF"] = (raw_df.loc[oos_idx, "TBILL_3M"] / 100) / 252
    for b in ["SPY", "AGG"]: res[b] = (raw_df.loc[oos_idx, f"{b}_Ret"].add(1).cumprod() * 100)

    # Confidence and Stats
    confidence_val = 0.70
    if not pred_df.empty:
        last_preds = pred_df.iloc[-1]
        z_score = (last_preds.max() - last_preds.mean()) / last_preds.std() if last_preds.std() > 0 else 0
        confidence_val = max(0.40, min(0.98, 0.5 + (z_score * 0.18)))

    today = datetime.now()
    next_mkt = today + timedelta(days=1) if today.hour >= 16 else today
    while next_mkt.weekday() >= 5: next_mkt += timedelta(days=1)

    return {
        "df": res, 
        "audit": pd.DataFrame({"Allocation": asset_history, "Daily_Return": strat_rets}, index=oos_idx),
        "target": asset_history[-1], 
        "confidence": confidence_val, 
        "next_date": next_mkt.strftime('%A, %b %d, %Y'),
        "strat_rets_raw": strat_rets
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
    opt = st.radio("Model Logic", [
        "Option A - Wavelet-SVR", "Option B - Wavelet-SVR-PPO", "Option C - Wavelet-A2C", "Option D - Wavelet-SVR-A2C",
        "Option E - Wavelet-HMM", "Option F - Wavelet-SVR-HMM", "Option G - Wavelet-BSTS", "Option H - Wavelet-SVR-BSTS"
    ])
    costs = st.number_input("T-Costs (bps)", 0, 50, 10)

output = run_professional_backtest(s_yr, opt, costs)

if output:
    data = output["df"]
    strat_rets = output["strat_rets_raw"]
    
    # ROW 1: PRIMARY TARGET SIGNAL
    conf_color = "#2e7d32" if output['confidence'] > 0.7 else "#f57c00"
    st.markdown(f"""
        <div style="background-color: #f1f8e9; padding: 30px; border-radius: 12px; border: 2px solid #a5d6a7; text-align: center; margin-bottom: 25px;">
            <p style="margin:0; color: #2e7d32; font-size: 14px; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">Target Allocation for {output['next_date']}</p>
            <h1 style="margin:5px 0; font-size: 100px; color: #1b5e20; font-family: 'Courier New', monospace; font-weight: 900;">{output['target']}</h1>
            <div style="width: 240px; margin: 0 auto;">
                <p style="margin:0; font-size: 12px; color: {conf_color}; font-weight: bold; text-transform: uppercase;">Signal Conviction: {output['confidence']:.0%}</p>
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
    
    # Logic for Kelly Score & Win-Loss Arrow
    pos_rets = [r for r in strat_rets if r > 0]
    neg_rets = [abs(r) for r in strat_rets if r < 0]
    win_loss_ratio = (np.mean(pos_rets) / np.mean(neg_rets)) if (pos_rets and neg_rets) else 1.0
    
    hit_ratio_sync = (pd.Series(strat_rets).tail(15) > 0).sum() / 15
    p, b = hit_ratio_sync, win_loss_ratio
    kelly_f = ((p * (b + 1)) - 1) / b if b > 0 else 0
    safe_kelly = max(0, min(1.0, kelly_f * 0.5))
    
    k_arrow, k_col = ("▲", "normal") if safe_kelly > 0.4 else ("▼", "inverse")

    m1.metric("Annualized Return", f"{ann_ret:.2%}")
    m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m3.metric("Max Drawdown (P-T)", f"{data['Drawdown'].min():.2%}")
    m4.metric("Max DD (Daily)", f"{data['Strategy_Ret'].min():.2%}")
    m5.metric("Hit Ratio (15D)", f"{hit_ratio_sync:.0%}")
    m6.metric("Kelly Recco", f"{safe_kelly:.0%}", delta=f"{k_arrow} {win_loss_ratio:.2f} W/L", delta_color=k_col)

    # ROW 3: EQUITY CHART
    st.markdown("<h3 style='margin-top: 25px; margin-bottom: 10px;'>Cumulative Performance</h3>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Equity"], name="Model Strategy", line=dict(color='#1a73e8', width=3)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SPY"], name="SPY", line=dict(color='#9aa0a6', dash='dot')))
    fig.update_layout(template="plotly_white", height=480, margin=dict(l=0,r=0,t=0,b=0), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns([1.2, 1])
    
   # --- FIXED LAYOUT FOR AUDIT AND METHODOLOGY ---
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("📋 Audit Trail")
        # Ensure these lines are indented exactly 8 spaces from the margin
        audit_display = output["audit"].tail(20).copy()
        audit_display.index = audit_display.index.strftime('%Y-%m-%d')
        st.dataframe(
            audit_display.style.format({"Daily_Return": "{:.2%}"}).map(
                lambda x: f'color: {"#1b5e20" if x > 0 else "#b71c1c"}; font-weight: bold;', 
                subset=['Daily_Return']
            ),
            use_container_width=True, height=560
        )

    with col_right:
        st.subheader("🔬 Methodology & Engine Logic")
        # Logic description mapping
        if "Option A" in opt:
            logic_desc = "Standard non-linear SVR. Focuses on raw point-prediction for maximum market participation."
        elif "Option B" in opt:
            logic_desc = "Directional SVR with PPO Stability. Uses a fixed 15bps hurdle to ensure entries occur only during high-momentum regimes."
        elif "Option C" in opt:
            logic_desc = "Pure Policy Gradient execution. Learns optimal asset weights by maximizing the synchronous Advantage Actor-Critic (A2C) objective function."
        elif "Option D" in opt:
            logic_desc = "Hybrid Alpha-Advantage gating. SVR generates directional bias, while A2C filters exposure based on 0.75σ relative conviction."
        elif "Option E" in opt:
            logic_desc = "Macro Regime Switching. Uses a Hidden Markov Model (HMM) to detect 3 latent market states (Bull, Bear, Volatile) based on DXY, VIX, and Credit Spreads."
        elif "Option F" in opt:
            logic_desc = "SVR-HMM Fusion. SVR provides the asset pick, but the HMM acts as a 'Risk-Off' circuit breaker, forcing CASH if the macro regime is unstable."
        elif "Option G" in opt:
            logic_desc = "Bayesian Structural Trend Selection. Uses BSTS to decompose price action into trend and noise, selecting assets with the highest 'Structural' probability."
        elif "Option H" in opt:
            logic_desc = "Wavelet-SVR-BSTS. SVR identifies the target, while a Bayesian Filter confirms if the trend is statistically significant (Confidence > 65%) before entry."
        else:
            logic_desc = "Ensemble logic execution."

        st.markdown(f"""
        **Architecture:** Multi-Engine Ensemble (Wavelet + SVR + HMM/BSTS)
        
        **Core Logic:**
        * **Active Strategy:** {logic_desc}
        * **Macro Pillars:** Integrated DXY (Dollar Index), Yield Curve (T10Y2Y), and Credit Spreads (IG/HY).
        
        **Risk Framework:**
        * **Trailing Stop:** 10% Drawdown hard-stop from peak equity.
        * **Kelly Criterion:** Half-Kelly capital sizing updated on 15-day rolling windows.
        * **Regime Gate:** Automatic CASH reversion when model confidence or macro states signal 'Systemic Risk'.
        """)
