import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from data.loader import load_raw_data
from engine import MomentumEngine, A2CEngine, DeepHybridEngine
from analytics.regime import RegimeHMM, BayesianFilter 

# --- CONFIG & THEME ---
st.set_page_config(page_title="P2 ETF WAVELET SVR MULTI MODEL", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; }
    [data-testid="stMetricValue"] { color: #1a73e8 !important; font-size: 22px !important; }
    [data-testid="stMetricLabel"] { font-weight: 700; text-transform: uppercase; font-size: 10px; color: #5f6368 !important; }
    </style>
""", unsafe_allow_html=True)

def get_next_trading_day_simple():
    now = datetime.now()
    if now.weekday() >= 4: 
        days_ahead = (7 - now.weekday()) if now.weekday() > 4 else 3
        next_day = now + timedelta(days=days_ahead)
    else:
        next_day = now + timedelta(days=1)
    return next_day.strftime('%d %B %Y')

@st.cache_data(ttl=3600, show_spinner=False)
def run_professional_backtest(start_yr, model_choice, t_costs_bps, stop_loss_pct, recovery_sigma, _force_sync=False):
    raw_df = load_raw_data(force_sync=_force_sync)
    assets = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
    
    for a in assets:
        if f"{a}_Ret" not in raw_df.columns:
            raw_df[f"{a}_Ret"] = raw_df[a].pct_change()
            
    t_cost_pct = t_costs_bps / 10_000
    from data.processor import build_feature_matrix

    all_preds = {}
    for ticker in assets:
        try:
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker)
            m_oos, m_is = idx.year >= start_yr, idx.year < start_yr
            
            # --- RECTIFIED MODEL ROUTING ---
            if any(opt in model_choice for opt in ["Option I", "Option J", "Option K"]):
                eng = DeepHybridEngine(mode=model_choice)
                oos_indices = np.where(m_oos)[0]
                X_3d = np.array([np.vstack([np.repeat(X[0:1], 20-len(X[max(0, i-19):i+1]), axis=0), X[max(0, i-19):i+1]]) for i in oos_indices])
                X_macro = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].loc[idx[m_oos]].values if "Option K" in model_choice else None
                preds = eng.predict_series(X_3d, X_macro=X_macro)

            elif "Option F" in model_choice:
                hmm = RegimeHMM()
                hmm.train_and_assign(raw_df.loc[idx[m_is]], assets)
                macro_oos = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].diff().loc[idx[m_oos]].fillna(0)
                preds = [1.0 if hmm.predict_best_asset(macro_oos.iloc[i:i+1]) == ticker else 0.0 for i in range(len(macro_oos))]

            elif "Option G" in model_choice: # SVR-HMM HYBRID (RECTIFIED)
                eng = MomentumEngine(); eng.train(X[m_is], y[m_is])
                hmm = RegimeHMM()
                hmm.train_and_assign(raw_df.loc[idx[m_is]], assets)
                macro_oos = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].diff().loc[idx[m_oos]].fillna(0)
                raw_svr = eng.predict_series(X[m_oos])
                preds = [raw_svr[i] if hmm.predict_best_asset(macro_oos.iloc[i:i+1]) == ticker else 0.0 for i in range(len(macro_oos))]

            elif "Option E" in model_choice: # INDEXING FIX
                bf = BayesianFilter()
                eng = MomentumEngine(); eng.train(X[m_is], y[m_is])
                conf_vec = bf.get_confidence(raw_df[ticker].loc[:idx[m_oos][-1]])
                preds = eng.predict_series(X[m_oos]) * conf_vec.iloc[-len(np.where(m_oos)[0]):].values

            elif "Option H" in model_choice:
                eng = MomentumEngine(); eng.train(X[m_is], y[m_is])
                bf = BayesianFilter()
                raw_preds = eng.predict_series(X[m_oos])
                conf_vec = bf.get_confidence(raw_df[ticker].loc[idx[m_oos]])
                preds = raw_preds * conf_vec.iloc[-len(raw_preds):].values

            elif "Option D" in model_choice:
                eng_s = MomentumEngine(); eng_s.train(X[m_is], y[m_is])
                eng_r = A2CEngine(); eng_r.train(X[m_is], y[m_is])
                preds = (eng_s.predict_series(X[m_oos]) + eng_r.predict_series(X[m_oos])) / 2

            elif "Option C" in model_choice:
                eng = A2CEngine(); eng.train(X[m_is], y[m_is])
                preds = eng.predict_series(X[m_oos])

            else: # Option A & B
                eng = MomentumEngine(); eng.train(X[m_is], y[m_is])
                preds = eng.predict_series(X[m_oos])
                
            all_preds[ticker] = pd.Series(preds, index=idx[m_oos])
        except: continue

    df_p = pd.DataFrame(all_preds).dropna()
    if df_p.empty: return None
    common_idx = df_p.index
    
    # --- EXECUTION LOOP (LOCKED STOP LOSS & DATA SYNC) ---
    equity, current_asset, hwm, in_timeout = 100.0, "CASH", 100.0, False
    rets, hist, confs = [], [], []

    for d in common_idx:
        hwm = max(hwm, equity)
        drawdown = (equity - hwm) / hwm
        if drawdown <= -stop_loss_pct: in_timeout = True 
        
        dp = df_p.loc[d]
        z_score = (dp.max() - dp.mean()) / dp.std() if dp.std() > 0 else 0
        raw_sig = dp.idxmax() if dp.max() > (0.0015 if "Option B" in model_choice else 0.0001) else "CASH"
        
        conf_display = 0.5 + (0.25 * (z_score / recovery_sigma)) 
        
        if in_timeout and z_score >= recovery_sigma: in_timeout = False
        final_sig = "CASH" if in_timeout else raw_sig
        
        if final_sig != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = final_sig
            
        day_r = (raw_df.loc[d, "TBILL_3M"]/100)/252 if current_asset == "CASH" else raw_df.loc[d, f"{current_asset}_Ret"]
        equity *= (1 + day_r)
        rets.append(day_r); hist.append(current_asset); confs.append(conf_display)

    res = pd.DataFrame(index=common_idx)
    res["Equity"] = (np.array(rets) + 1).cumprod() * 100
    res["Strategy_Ret"] = rets
    res["Drawdown"] = (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    res["RF"] = (raw_df.loc[common_idx, "TBILL_3M"] / 100) / 252
    
    for b in ["SPY", "AGG"]:
        res[b] = (raw_df.loc[common_idx, f"{b}_Ret"] + 1).cumprod() * 100
    
    return {"df": res, "audit": pd.DataFrame({"Allocation": hist, "Return": rets, "Z-Score": confs}, index=common_idx), 
            "target": hist[-1], "conf": confs[-1], "date": common_idx[-1].strftime('%Y-%m-%d')}

# --- SIDEBAR ---
with st.sidebar:
    st.header("Terminal Config")
    if st.button("🔄 Refresh Data & Cache"):
        st.cache_data.clear(); st.rerun()
    
    s_yr = st.slider("Backtest Start Year", 2010, 2024, 2015)
    opt = st.radio("Intelligence Engine", [
        "Option A- Wavelet-SVR", "Option B-Wavelet-SVR-PPO", "Option C: Wavelet-A2C", 
        "Option D: Wavelet-SVR-A2C", "Option E: Wavelet-Bayesian-Regime", "Option F: Wavelet-HMM", 
        "Option G- Wavelet-SVR-HMM", "Option H: Wavelet-SVR-Bayesian", "Option I: Wavelet- CNN-LSTM", 
        "Option J: Wavelet-Attention-CNN-LSTM", "Option K- Wavelet- Parallel-Dual-Stream-CNN-LSTM"
    ])
    
    st.subheader("Risk Gate Controls")
    sl_input = st.slider("Trailing Stop Loss (%)", 8.0, 20.0, 18.0) / 100
    rec_sigma = st.slider("Recovery Threshold (Sigma)", 1.0, 2.0, 1.4, 0.1)
    costs = st.number_input("T-Costs (bps)", 0, 50, 10)

# --- EXECUTE ---
out = run_professional_backtest(s_yr, opt, costs, sl_input, rec_sigma)
if out is None:
    st.error("IndexError: No valid prediction data. Adjust Start Year.")
else:
    df = out["df"]

    # --- UI: HEADER ---
    st.title("P2 Wavelet Multi-Model")
    st.markdown(f"""
        <div style="background-color: #f1f8e9; padding: 25px; border-radius: 15px; border: 2px solid #a5d6a7; text-align: center; margin-bottom: 25px;">
            <p style="margin:0; color: #2e7d32; font-size: 14px; font-weight: 700; text-transform: uppercase;">Prediction for NYSE Trading Date: {get_next_trading_day_simple()}</p>
            <h1 style="margin:5px 0; font-size: 90px; color: #1b5e20; line-height: 1;">{out['target']}</h1>
            <p style="margin:0; font-size: 20px; color: #388e3c; font-weight: 500;">Scaled Conviction: {out['conf']:.1%}</p>
        </div>
    """, unsafe_allow_html=True)

    # --- UI: METRICS ---
    ann_ret = (df["Equity"].iloc[-1] / 100) ** (252 / len(df)) - 1
    sharpe = ((df["Strategy_Ret"] - df["RF"]).mean() / df["Strategy_Ret"].std()) * np.sqrt(252)
    max_dd_pt = df["Drawdown"].min()
    hit_ratio = (df["Strategy_Ret"].tail(15) > 0).mean()
    ann_vol = df["Strategy_Ret"].std() * np.sqrt(252)
    kelly = 0.5 * (ann_ret / (ann_vol**2)) if ann_vol > 0 and ann_ret > 0 else 0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Annual Return", f"{ann_ret:.2%}")
    m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m3.metric("Max DD (P/T)", f"{max_dd_pt:.2%}")
    m4.metric("Volatility", f"{ann_vol:.2%}")
    m5.metric("Hit Ratio (15D)", f"{hit_ratio:.0%}")
    m6.metric("1/2 Kelly", f"{min(kelly, 0.5):.1%}" if out['target'] != "CASH" else "0.0%")

    # --- UI: WORST DAILY EVENT (RESTORED) ---
    worst_val = df["Strategy_Ret"].min()
    worst_date = df["Strategy_Ret"].idxmin().strftime('%Y-%m-%d')
    st.markdown(f"""
        <div style="border: 1px solid #ffcdd2; background-color: #ffebee; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 20px;">
            <span style="color: #c62828; font-weight: bold;">Worst Daily Event: {worst_val:.2%} drawdown occurred on {worst_date}</span>
        </div>
    """, unsafe_allow_html=True)

    # --- UI: CHARTS ---
    st.subheader("OOS Cumulative Return")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity"], name="P2 Strategy", line=dict(color='#1a73e8', width=3)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SPY"], name="SPY Bench", line=dict(color='#718096', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df["AGG"], name="AGG Bench", line=dict(color='#e53e3e', dash='dot')))
    fig.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1, x=1, xanchor='right'), height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- UI: AUDIT (PROFESSIONAL COLOR CODING) ---
    st.subheader("15-Day Allocation Audit Trail")
    audit_df = out["audit"].tail(15).copy()
    audit_df.index = audit_df.index.strftime('%Y-%m-%d')

    def color_rets(val):
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'

    st.dataframe(audit_df.style.applymap(color_rets, subset=['Return']).format({'Return': '{:.2%}', 'Z-Score': '{:.2f}'}), use_container_width=True)

    # --- METHODOLOGY ---
    methodologies = {
        "Option A": "MODWT multi-resolution analysis combined with Polynomial SVR.",
        "Option B": "Hybrid RL-Supervised model using PPO for high-probability entry.",
        "Option C": "Advantage Actor-Critic (A2C) optimizing allocation.",
        "Option D": "SVR-A2C Ensemble weighting predictions.",
        "Option E": "Bayesian state-space filtering for defensive regime detection.",
        "Option F": "Hidden Markov Model (HMM) for latent regime classification.",
        "Option G": "HMM-Biased SVR: SVR magnitude weight applied to HMM asset selection.",
        "Option H": "Bayesian-Denoised SVR applying shrinkage priors.",
        "Option I": "CNN-LSTM Deep Learning for spatial and temporal features.",
        "Option J": "Attention-Augmented CNN-LSTM focusing on frequency bands.",
        "Option K": "Parallel Dual-Stream Deep Fusion with macro-economic vectors."
    }
    method_key = opt.split("-")[0].strip() if "-" in opt else opt.split(":")[0].strip()
    st.divider()
    st.markdown(f"### Methodology: {opt}")
    st.write(methodologies.get(method_key, "Wavelet-based multi-resolution analysis."))
    st.info(f"⚠️ **Risk Policy:** Trailing Stop Loss at {sl_input:.1%}. Recovery requires Z-Score > {rec_sigma}.")
