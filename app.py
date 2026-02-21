import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data.loader import load_raw_data
from engine import MomentumEngine, A2CEngine, DeepHybridEngine 

# --- CONFIG & THEME ---
st.set_page_config(page_title="P2 ETF WAVELET SVR MULTI MODEL", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; }
    [data-testid="stMetricValue"] { color: #1a73e8 !important; font-size: 28px !important; }
    [data-testid="stMetricLabel"] { font-weight: 700; text-transform: uppercase; font-size: 11px; color: #5f6368 !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def run_professional_backtest(start_yr, model_choice, t_costs_bps):
    raw_df = load_raw_data()
    assets = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
    t_cost_pct = t_costs_bps / 10_000
    from data.processor import build_feature_matrix

    all_preds = {}
    oos_mask_global = raw_df.index.year >= start_yr
    is_mask_global = raw_df.index.year < start_yr

    # 1. PREDICTION ENGINE LOOP
    for ticker in assets:
        try:
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker)
            m_oos, m_is = idx.year >= start_yr, idx.year < start_yr
            
            if any(opt in model_choice for opt in ["Option I", "Option J", "Option K"]):
                eng = DeepHybridEngine(mode=model_choice)
                X_vals = X.values
                oos_indices = np.where(m_oos)[0]
                # Build 3D Tensor for Deep Learning (20 Day Lookback)
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
    common_idx = raw_df.index[oos_mask_global]
    
    # 2. EXECUTION LOOP WITH 12% HWM RISK GATE
    equity, current_asset, hwm, in_timeout = 100.0, "CASH", 100.0, False
    rets, hist, confs = [], [], []

    for d in common_idx:
        hwm = max(hwm, equity)
        drawdown = (equity - hwm) / hwm
        if drawdown <= -0.12: in_timeout = True
        
        dp = df_p.loc[d] if d in df_p.index else None
        # Option B applies the 0.0015 PPO-optimized noise threshold
        raw_sig = dp.idxmax() if (dp is not None and dp.max() > (0.0015 if "Option B" in model_choice else 0)) else "CASH"
        conf = 0.5 + (0.18 * ((dp.max()-dp.mean())/dp.std())) if (dp is not None and dp.std()>0) else 0.5
        
        # Risk Gate Re-entry: requires 75% Conviction
        if in_timeout and conf >= 0.75: in_timeout = False
        final_sig = "CASH" if in_timeout else raw_sig
        
        if final_sig != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = final_sig
            
        day_r = (raw_df.loc[d, "TBILL_3M"]/100)/252 if current_asset == "CASH" else raw_df.loc[d, f"{current_asset}_Ret"]
        equity *= (1 + day_r)
        rets.append(day_r); hist.append(current_asset); confs.append(conf)

    res = pd.DataFrame({"Equity": (pd.Series(rets).add(1).cumprod()*100).values}, index=common_idx)
    res["Strategy_Ret"] = rets
    res["Drawdown"] = (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    res["RF"] = (raw_df.loc[common_idx, "TBILL_3M"] / 100) / 252
    for b in ["SPY", "AGG"]: res[b] = (raw_df.loc[common_idx, f"{b}_Ret"].add(1).cumprod()*100)
    
    return {"df": res, "audit": pd.DataFrame({"Allocation": hist, "Return": rets}, index=common_idx), "target": hist[-1], "conf": confs[-1], "date": common_idx[-1].strftime('%Y-%m-%d')}

# --- SIDEBAR ---
with st.sidebar:
    st.header("Terminal Config")
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

# --- UI: PRIMARY TARGET & CONVICTION ---
st.markdown(f"""
    <div style="background-color: #f1f8e9; padding: 25px; border-radius: 15px; border: 2px solid #a5d6a7; text-align: center; margin-bottom: 25px;">
        <p style="margin:0; color: #2e7d32; font-size: 14px; font-weight: 700; text-transform: uppercase;">Next Trading Session Target</p>
        <h1 style="margin:5px 0; font-size: 90px; color: #1b5e20; line-height: 1;">{out['target']}</h1>
        <p style="margin:0; font-size: 20px; color: #388e3c; font-weight: 500;">Signal Conviction: {out['conf']:.1%}</p>
    </div>
""", unsafe_allow_html=True)

# --- UI: METRIC CALCULATIONS ---
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

st.markdown(f"<p style='font-size: 13px; color: #666; margin-top: -10px;'><b>Max Drawdown Event:</b> {max_dd_val:.2%} occurred on <b>{max_dd_date}</b></p>", unsafe_allow_html=True)

# --- UI: CHARTS ---
st.subheader("OOS Cumulative Return (Normalized to 100)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Equity"], name="P2 Strategy", line=dict(color='#1a73e8', width=3)))
fig.add_trace(go.Scatter(x=df.index, y=df["SPY"], name="SPY Bench", line=dict(color='rgba(100,100,100,0.4)', dash='dot')))
fig.add_trace(go.Scatter(x=df.index, y=df["AGG"], name="AGG Bench", line=dict(color='rgba(180,0,0,0.4)', dash='dot')))
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1, x=1, xanchor='right'), height=500)
st.plotly_chart(fig, use_container_width=True)

# --- UI: AUDIT ---
st.subheader("15-Day Allocation Audit Trail")
audit_df = out["audit"].tail(15).copy()
audit_df.index = audit_df.index.strftime('%Y-%m-%d')
st.dataframe(audit_df.style.format({'Return': '{:.2%}'}), use_container_width=True)

# --- UI: METHODOLOGY ---
methodologies = {
    "Option A": "MODWT Wavelet denoising + Polynomial SVR for non-linear momentum.",
    "Option B": "SVR signal + PPO agent risk-overlay filtering via adaptive threshold.",
    "Option C": "Advantage Actor-Critic (RL) for continuous allocation policy.",
    "Option D": "Hybrid SVR-A2C weighting or HMM regime detection.",
    "Option E": "Bayesian state-space filtering for probabilistic regime transition.",
    "Option G": "HMM regime biasing for the SVR momentum engine.",
    "Option H": "SVR + Bayesian noise reduction for high-conviction entries.",
    "Option I": "3D CNN-LSTM architecture on wavelet temporal tensors.",
    "Option J": "CNN-LSTM + Spatial-Temporal Attention for frequency band prioritization.",
    "Option K": "Parallel Dual-Stream Network fusing price and macro vectors (VIX/DXY)."
}
method_key = opt.split("-")[0].strip() if "-" in opt else opt.split(":")[0].strip()
st.divider()
st.markdown(f"### Methodology: {opt}")
st.write(methodologies.get(method_key, "Wavelet-based multi-resolution analysis."))
st.info("⚠️ **Risk Gate:** Force-CASH triggered at 12% drawdown from HWM. Re-entry requires Signal Conviction > 75%.")
