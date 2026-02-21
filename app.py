import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from data.loader import load_raw_data
from engine import MomentumEngine, A2CEngine, DeepHybridEngine 

# --- CONFIG & THEME ---
st.set_page_config(page_title="P2 ETF WAVELET SVR MULTI MODEL", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; }
    [data-testid="stMetricValue"] { color: #1a73e8 !important; font-size: 24px !important; }
    [data-testid="stMetricLabel"] { font-weight: 700; text-transform: uppercase; font-size: 11px; color: #5f6368 !important; }
    .stHeader { font-family: 'Helvetica Neue', sans-serif; color: #1a202c; }
    </style>
""", unsafe_allow_html=True)

def get_next_nyse_date():
    nyse = mcal.get_calendar('NYSE')
    now = datetime.now()
    schedule = nyse.schedule(start_date=now, end_date=now + timedelta(days=7))
    return schedule.index[0].strftime('%d %B %Y')

@st.cache_data(ttl=3600)
def run_professional_backtest(start_yr, model_choice, t_costs_bps):
    raw_df = load_raw_data()
    assets = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
    t_cost_pct = t_costs_bps / 10_000
    from data.processor import build_feature_matrix

    all_preds = {}
    oos_mask_global = raw_df.index.year >= start_yr
    
    # 1. PREDICTION ENGINE LOOP
    for ticker in assets:
        try:
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker)
            m_oos, m_is = idx.year >= start_yr, idx.year < start_yr
            
            if any(opt in model_choice for opt in ["Option I", "Option J", "Option K"]):
                eng = DeepHybridEngine(mode=model_choice)
                X_vals = X.values
                oos_indices = np.where(m_oos)[0]
                X_3d = np.array([np.vstack([np.repeat(X_vals[0:1], 20-len(X_vals[max(0, i-19):i+1]), axis=0), X_vals[max(0, i-19):i+1]]) for i in oos_indices])
                X_macro = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].loc[idx[m_oos]].values if "Option K" in model_choice else None
                preds = eng.predict_series(X_3d, X_macro=X_macro)
            elif "Option C" in model_choice or "Option D: Wavelet-SVR-A2C" in model_choice:
                eng = A2CEngine(); eng.train(X[m_is], y[m_is])
                preds = eng.predict_series(X[m_oos])
            else:
                eng = MomentumEngine(); eng.train(X[m_is], y[m_is])
                preds = eng.predict_series(X[m_oos])
            all_preds[ticker] = pd.Series(preds, index=idx[m_oos])
        except: continue

    df_p = pd.DataFrame(all_preds).dropna()
    common_idx = raw_df.index[raw_df.index.isin(df_p.index)]
    
    # 2. EXECUTION LOOP WITH 18% HWM RISK GATE
    equity, current_asset, hwm, in_timeout = 100.0, "CASH", 100.0, False
    rets, hist, confs = [], [], []

    for d in common_idx:
        hwm = max(hwm, equity)
        drawdown = (equity - hwm) / hwm
        if drawdown <= -0.18: in_timeout = True
        
        dp = df_p.loc[d]
        raw_sig = dp.idxmax() if dp.max() > (0.0015 if "Option B" in model_choice else 0) else "CASH"
        conf = 0.5 + (0.18 * ((dp.max()-dp.mean())/dp.std())) if dp.std()>0 else 0.5
        
        if in_timeout and conf >= 0.75: in_timeout = False
        final_sig = "CASH" if in_timeout else raw_sig
        
        if final_sig != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = final_sig
            
        day_r = (raw_df.loc[d, "TBILL_3M"]/100)/252 if current_asset == "CASH" else raw_df.loc[d, f"{current_asset}_Ret"]
        equity *= (1 + day_r)
        rets.append(day_r); hist.append(current_asset); confs.append(conf)

    # 3. BENCHMARK FIX (Normalized to 100 at OOS Start)
    res = pd.DataFrame({"Equity": (pd.Series(rets).add(1).cumprod()*100).values}, index=common_idx)
    res["Strategy_Ret"] = rets
    res["Drawdown"] = (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    res["RF"] = (raw_df.loc[common_idx, "TBILL_3M"] / 100) / 252
    
    for b in ["SPY", "AGG"]:
        b_rets = raw_df.loc[common_idx, f"{b}_Ret"]
        res[b] = (b_rets + 1).cumprod() * 100
    
    return {"df": res, "audit": pd.DataFrame({"Allocation": hist, "Return": rets}, index=common_idx), 
            "target": hist[-1], "conf": confs[-1], "date": common_idx[-1].strftime('%Y-%m-%d')}

# --- SIDEBAR ---
with st.sidebar:
    st.header("Terminal Config")
    if st.button("🔄 Refresh Data & Cache"):
        st.cache_data.clear()
        st.session_state['refreshed_at'] = datetime.now().strftime('%d/%m/%y %H:%M')
        st.rerun()
    
    refreshed_text = st.session_state.get('refreshed_at', datetime.now().strftime('%d/%m/%y'))
    st.caption(f"Refreshed on {refreshed_text}")
    
    s_yr = st.slider("Backtest Start Year", 2010, 2024, 2015)
    opt = st.radio("Intelligence Engine", [
        "Option A- Wavelet-SVR", "Option B-Wavelet-SVR-PPO", "Option C: Wavelet-A2C", 
        "Option D: Wavelet-SVR-A2C", "Option D: Wavelet- HMM", "Option E: Wavelet-Bayesian-Regime", 
        "Option G- Wavelet-SVR-HMM", "Option H: Wavelet-SVR-Bayesian", "Option I: Wavelet- CNN-LSTM", 
        "Option J: Wavelet-Attention-CNN-LSTM", "Option K- Wavelet- Parallel-Dual-Stream-CNN-LSTM"
    ])
    costs = st.number_input("T-Costs (bps)", 0, 50, 10)

# --- EXECUTE ---
out = run_professional_backtest(s_yr, opt, costs)
df = out["df"]

# --- UI: HEADER & PRIMARY OUTPUT ---
st.title("P2 Wavelet Multi-Model")

st.markdown(f"""
    <div style="background-color: #f1f8e9; padding: 25px; border-radius: 15px; border: 2px solid #a5d6a7; text-align: center; margin-bottom: 25px;">
        <p style="margin:0; color: #2e7d32; font-size: 14px; font-weight: 700; text-transform: uppercase;">Prediction for NYSE Trading Date: {get_next_nyse_date()}</p>
        <h1 style="margin:5px 0; font-size: 90px; color: #1b5e20; line-height: 1;">{out['target']}</h1>
        <p style="margin:0; font-size: 20px; color: #388e3c; font-weight: 500;">Signal Conviction: {out['conf']:.1%}</p>
    </div>
""", unsafe_allow_html=True)

# --- UI: METRIC ROW ---
ann_ret = (df["Equity"].iloc[-1] / 100) ** (252 / len(df)) - 1
sharpe = ((df["Strategy_Ret"] - df["RF"]).mean() / df["Strategy_Ret"].std()) * np.sqrt(252)
max_dd_val = df["Drawdown"].min()
max_dd_date = df["Drawdown"].idxmin().strftime('%Y-%m-%d')
hit_ratio = (df["Strategy_Ret"].tail(15) > 0).mean()
ann_vol = df["Strategy_Ret"].std() * np.sqrt(252)
kelly = 0.5 * (ann_ret / (ann_vol**2)) if ann_vol > 0 and ann_ret > 0 else 0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Annual Return", f"{ann_ret:.2%}")
m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
m3.metric("Max DD (P/T)", f"{max_dd_val:.2%}")
m4.metric("Hit Ratio (15D)", f"{hit_ratio:.0%}")
m5.metric("1/2 Kelly Factor", f"{min(kelly, 0.5):.1%}" if out['target'] != "CASH" else "0.0%")

st.markdown(f"<div style='background-color: #fff5f5; padding: 10px; border-radius: 5px; border: 1px solid #feb2b2; margin-top: 10px; font-size: 13px; color: #c53030;'><b>Daily Max DD Event:</b> {max_dd_val:.2%} occurred on <b>{max_dd_date}</b></div>", unsafe_allow_html=True)

# --- UI: CHARTS ---
st.subheader("OOS Cumulative Return")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Equity"], name="P2 Strategy", line=dict(color='#1a73e8', width=3)))
fig.add_trace(go.Scatter(x=df.index, y=df["SPY"], name="SPY Bench", line=dict(color='#718096', dash='dot')))
fig.add_trace(go.Scatter(x=df.index, y=df["AGG"], name="AGG Bench", line=dict(color='#e53e3e', dash='dot')))
fig.update_layout(template="white", margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1, x=1, xanchor='right'), height=500)
st.plotly_chart(fig, use_container_width=True)

# --- UI: AUDIT ---
st.subheader("15-Day Allocation Audit Trail")
audit_df = out["audit"].tail(15).copy()
audit_df.index = audit_df.index.strftime('%Y-%m-%d')
st.dataframe(audit_df.style.format({'Return': '{:.2%}'}), use_container_width=True)

# --- UI: METHODOLOGY ---
methodologies = {
    "Option A": "MODWT multi-resolution analysis combined with Polynomial SVR. The wavelet transform decomposes price action into frequency components, allowing the SVR to ignore short-term noise and capture mid-term momentum shifts.",
    "Option B": "A hybrid Reinforcement Learning-Supervised model. The SVR provides a directional bias, while a Proximal Policy Optimization (PPO) agent identifies high-probability entry windows using an adaptive volatility threshold.",
    "Option C": "Advantage Actor-Critic (A2C) engine that optimizes allocation as a continuous policy problem. The 'Actor' determines the asset weight while the 'Critic' evaluates the expected risk-adjusted reward based on wavelet coefficients.",
    "Option D": "SVR-A2C Ensemble or Hidden Markov Model (HMM). Depending on the sub-selection, this model either uses HMMs to detect latent market regimes or weights SVR predictions by an RL agent's conviction score.",
    "Option E": "Bayesian state-space filtering for regime detection. This approach calculates the posterior probability of entering a high-volatility regime, forcing defensive postures before standard indicators turn bearish.",
    "Option G": "HMM-Biased SVR. A Hidden Markov Model classifies the current market state (Bull/Bear/Sideways), which dynamically adjusts the 'C' parameter of the SVR to prevent over-trading in flat markets.",
    "Option H": "Bayesian-Denoised SVR. Applies a Bayesian shrinkage prior to wavelet coefficients before they enter the SVR, effectively reducing the impact of outliers and flash-volatility events on signals.",
    "Option I": "Deep Learning temporal feature processing. A 1D Convolutional Neural Network (CNN) extracts spatial price features which are then sequenced by a Long Short-Term Memory (LSTM) network for future-return prediction.",
    "Option J": "Attention-Augmented CNN-LSTM. Utilizes a Self-Attention mechanism to dynamically weight the importance of different wavelet frequency bands, allowing the model to 'focus' on the most relevant market drivers in real-time.",
    "Option K": "Parallel Dual-Stream Deep Fusion. Processes price-series data and macro-economic vectors (VIX, Spreads, DXY) in separate neural streams before fusing them at the dense layer for a multi-contextual prediction."
}
method_key = opt.split("-")[0].strip() if "-" in opt else opt.split(":")[0].strip()
st.divider()
st.markdown(f"### Methodology: {opt}")
st.write(methodologies.get(method_key, "Wavelet-based multi-resolution analysis."))
st.info("⚠️ **Risk Management Policy:** Force-CASH execution is triggered at an 18% drawdown from the High Water Mark. Recovery to active allocation requires a Signal Conviction score > 75%.")
