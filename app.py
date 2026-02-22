import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.loader import load_raw_data
from engine import MomentumEngine, A2CEngine, DeepHybridEngine
from analytics.regime import RegimeHMM, BayesianFilter 

# --- CONFIG & THEME ---
st.set_page_config(page_title="P2 ETF WAVELET SVR MULTI MODEL", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; }
    [data-testid="stMetricValue"] { color: #1a73e8 !important; font-size: 24px !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-weight: 700; text-transform: uppercase; font-size: 11px; color: #5f6368 !important; }
    .metric-sub { font-size: 12px; color: #70757a; margin-top: -12px; font-weight: 500; }
    .metric-sub b { color: #d93025; }
    </style>
""", unsafe_allow_html=True)

def get_next_trading_day_simple():
    now = datetime.now()
    days_ahead = (7 - now.weekday()) if now.weekday() >= 4 else 1
    next_day = now + timedelta(days=days_ahead)
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
            m_oos = idx.year >= start_yr
            m_is = idx.year < start_yr
            
            if any(opt in model_choice for opt in ["Option I", "Option J", "Option K"]):
                eng = DeepHybridEngine(mode=model_choice)
                # Map the UI choice to the correct file you saved
                model_map = {"Option I": "opt_i_cnn.h5", "Option J": "opt_j_cnn_lstm.h5", "Option K": "opt_k_hybrid.h5"}
                eng.load(f"models/{model_map[model_choice]}")
                oos_indices = np.where(m_oos)[0]
                X_3d = np.array([np.vstack([np.repeat(X[0:1], 20-len(X[max(0, i-19):i+1]), axis=0), X[max(0, i-19):i+1]]) for i in oos_indices])
                X_macro = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].loc[idx[m_oos]].values if "Option K" in model_choice else None
                preds = eng.predict_series(X_3d, X_macro=X_macro)
            # Move HMM training ABOVE the 'for ticker in assets' loop to save time
                hmm_model = None
            if "Option F" in model_choice or "Option G" in model_choice:
        try:
            hmm_model = RegimeHMM()
            hmm_model.train_and_assign(raw_df.loc[idx[m_is]], assets)
        except:
            hmm_model = None # Fallback if training fails

    for ticker in assets:
        try:
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker)
            # ... (other logic) ...

            elif "Option F" in model_choice:
                if hmm_model:
                    macro_oos = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].diff().loc[idx[m_oos]].fillna(0)
                    preds = [1.0 if hmm_model.predict_best_asset(macro_oos.iloc[i:i+1]) == ticker else 0.0 for i in range(len(macro_oos))]
                else:
                    preds = np.zeros(np.sum(m_oos)) # Safety fallback

            elif "Option G" in model_choice:
                eng = MomentumEngine()
                eng.load("models/svr_momentum_poly.pkl")
                raw_svr = eng.predict_series(X[m_oos])
                if hmm_model:
                    macro_oos = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].diff().loc[idx[m_oos]].fillna(0)
                    preds = [raw_svr[i] * 1.15 if hmm_model.predict_best_asset(macro_oos.iloc[i:i+1]) == ticker else 0.0 for i in range(len(macro_oos))]
                else:
                    preds = raw_svr # Fallback to pure SVR if HMM fails
            elif "Option H" in model_choice:
                # Option H uses SVR + Bayesian, so we LOAD the SVR weights
                eng = MomentumEngine()
                eng.load("models/svr_momentum_poly.pkl")
                bf = BayesianFilter()
                conf_vec = bf.get_confidence(raw_df[ticker].loc[:idx[m_oos][-1]])
                preds = eng.predict_series(X[m_oos]) * conf_vec.values[-np.sum(m_oos):]
                
            elif "Option E" in model_choice:
                # Option E is just Bayesian-Regime; no SVR engine needed
                bf = BayesianFilter()
                # Assuming Option E uses the confidence score as the direct prediction
                preds = bf.get_confidence(raw_df[ticker].loc[:idx[m_oos][-1]]).values[-np.sum(m_oos):]
            elif "Option C" in model_choice:
                eng = A2CEngine()
                eng.load("models/a2c_weights.pkl") # Ensure your trainer saves this name
                preds = eng.predict_series(X[m_oos])
            else:
                eng = MomentumEngine()
                eng.load("models/svr_momentum_poly.pkl") # Use the pre-trained file
                preds = eng.predict_series(X[m_oos])
            all_preds[ticker] = pd.Series(preds, index=idx[m_oos])
        except Exception:
            continue

    df_p = pd.DataFrame(all_preds).dropna()
    if df_p.empty: return None
    common_idx = df_p.index
    
    equity, current_asset, hwm, in_timeout = 100.0, "CASH", 100.0, False
    rets, hist, confs = [], [], []

    for d in common_idx:
        dp = df_p.loc[d]
        z_score = (dp.max() - dp.mean()) / dp.std() if dp.std() > 0 else 0
        
        hwm = max(hwm, equity)
        drawdown = (equity - hwm) / hwm
        if drawdown <= -stop_loss_pct: in_timeout = True 
        if in_timeout and z_score >= recovery_sigma: in_timeout = False
        
        raw_sig = dp.idxmax() if dp.max() > 0.0001 else "CASH"
        final_sig = "CASH" if in_timeout else raw_sig
        
        if final_sig != current_asset:
            equity *= (1 - t_cost_pct); current_asset = final_sig
            
        day_r = (raw_df.loc[d, "TBILL_3M"]/100)/252 if current_asset == "CASH" else raw_df.loc[d, f"{current_asset}_Ret"]
        equity *= (1 + day_r)
        rets.append(day_r); hist.append(current_asset); confs.append(z_score)

   # --- FINAL RECTIFIED DATA ALIGNMENT (ALL METRICS + GRAPH) ---
    res = pd.DataFrame(index=common_idx)
    res.index = pd.to_datetime(res.index)
    res["Equity"] = (np.array(rets) + 1).cumprod() * 100
    res["Strategy_Ret"] = rets
    
    # Calculate Drawdown for the c3/c4 metrics
    res["Drawdown"] = (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    
    # Restore RF for Sharpe Calculation
    res["RF"] = (raw_df.loc[common_idx, "TBILL_3M"] / 100) / 252
    
    # Map Benchmarks
    res["SPY"] = (raw_df.loc[common_idx, "SPY_Ret"] + 1).cumprod() * 100
    res["AGG"] = (raw_df.loc[common_idx, "AGG_Ret"] + 1).cumprod() * 100
    
    res = res.dropna()
    
    return {"df": res, "audit": pd.DataFrame({"Allocation": hist, "Return": rets, "Z-Score": confs}, index=common_idx), 
            "target": hist[-1], "conf": confs[-1], "date": common_idx[-1].strftime('%Y-%m-%d')}

# --- SIDEBAR ---
with st.sidebar:
    st.header("Terminal Config")
    if st.button("🔄 Refresh Data & Cache"):
        st.cache_data.clear()
        raw_data = load_raw_data()
        st.toast(f"Data Synced: {raw_data.index[-1].strftime('%Y-%m-%d')}")
        st.rerun()
    
    s_yr = st.slider("Backtest Start Year", 2010, 2024, 2015)
    opt = st.radio("Intelligence Engine", ["Option A- Wavelet-SVR", "Option B-Wavelet-SVR-PPO", "Option C: Wavelet-A2C", "Option D: Wavelet-SVR-A2C", "Option E: Wavelet-Bayesian-Regime", "Option F: Wavelet-HMM", "Option G- Wavelet-SVR-HMM", "Option H: Wavelet-SVR-Bayesian", "Option I: Wavelet- CNN-LSTM", "Option J: Wavelet-Attention-CNN-LSTM", "Option K- Wavelet- Parallel-Dual-Stream-CNN-LSTM"])
    st.subheader("Risk Gate Controls")
    sl_input = st.slider("Trailing Stop Loss (%)", 8.0, 20.0, 18.0, 0.5) / 100
    rec_sigma = st.slider("Recovery Threshold (Sigma)", 1.0, 2.0, 1.4, 0.1)
    costs = st.number_input("T-Costs (bps)", 0, 50, 10)

# --- UI EXECUTION ---
out = run_professional_backtest(s_yr, opt, costs, sl_input, rec_sigma)
if out:
    df = out["df"]
    st.title("P2 Wavelet Multi-Model")
    
    st.markdown(f"""<div style="background-color: #f1f8e9; padding: 25px; border-radius: 15px; border: 2px solid #a5d6a7; text-align: center; margin-bottom: 25px;"><p style="margin:0; color: #2e7d32; font-size: 14px; font-weight: 700; text-transform: uppercase;">Prediction for NYSE: {get_next_trading_day_simple()}</p><h1 style="margin:5px 0; font-size: 90px; color: #1b5e20; line-height: 1;">{out['target']}</h1><p style="margin:0; font-size: 20px; color: #388e3c; font-weight: 500;">Current Z-Score: {out['conf']:.2f}σ</p></div>""", unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    ann_ret = (df["Equity"].iloc[-1] / 100) ** (252 / len(df)) - 1
    c1.metric("Annual Return", f"{ann_ret:.2%}")
    sharpe = ((df['Strategy_Ret']-df['RF']).mean()/df['Strategy_Ret'].std())*np.sqrt(252)
    c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c3.metric("Max DD (P/T)", f"{df['Drawdown'].min():.2%}")
    with c4:
        st.metric("Max DD (Daily)", f"{df['Strategy_Ret'].min():.2%}")
        st.markdown(f'<p class="metric-sub">Worst: <b>{df["Strategy_Ret"].min():.2%}</b> on {df["Strategy_Ret"].idxmin().strftime("%Y-%m-%d")}</p>', unsafe_allow_html=True)
    with c5:
        hit_ratio_15d = (df["Strategy_Ret"].tail(15) > 0).mean()
        st.metric("Hit Ratio (15D)", f"{hit_ratio_15d:.1%}")
        st.markdown(f'<p class="metric-sub">Last 15 Trading Sessions</p>', unsafe_allow_html=True)

    st.subheader("OOS Cumulative Return")
    fig = go.Figure()

    # Explicitly map x to df.index to stop the diagonal line
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df["Equity"], 
        name="P2 Strategy", 
        line=dict(color='#1a73e8', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df["SPY"], 
        name="SPY Bench", 
        line=dict(color='#718096', dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df["AGG"], 
        name="AGG Bench", 
        line=dict(color='#e53e3e', dash='dot')
    ))

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(type='date', tickformat='%Y-%m'), # Force Date formatting
        height=500,
        margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.1, x=1, xanchor='right')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("15-Day Audit Trail")
    audit_df = out["audit"].tail(15).copy()
    audit_df.index = audit_df.index.strftime('%Y-%m-%d')
    st.dataframe(audit_df.style.applymap(lambda v: 'color: #d93025' if isinstance(v, (int, float)) and v < 0 else 'color: #188038', subset=['Return']).format({'Return': '{:.2%}', 'Z-Score': '{:.2f}'}), use_container_width=True)

    # --- FULL METHODOLOGY FOOTER ---
    methodologies = {
        "Option A": "MODWT multi-resolution analysis combined with Polynomial SVR. Wavelet transform captures mid-term momentum shifts.",
        "Option B": "Hybrid RL-Supervised model using PPO for high-probability entry windows.",
        "Option C": "Advantage Actor-Critic (A2C) optimizing allocation as a continuous policy.",
        "Option D": "SVR-A2C Ensemble weighting predictions by agent conviction scores.",
        "Option E": "Bayesian state-space filtering for defensive regime detection.",
        "Option F": "Hidden Markov Model (HMM) for latent regime classification.",
        "Option G": "HMM-Biased SVR that adjusts conviction parameters based on market state.",
        "Option H": "Bayesian-Denoised SVR applying shrinkage priors to wavelet coefficients.",
        "Option I": "CNN-LSTM Deep Learning for spatial and temporal feature extraction.",
        "Option J": "Attention-Augmented CNN-LSTM focusing on relevant frequency bands.",
        "Option K": "Parallel Dual-Stream Deep Fusion incorporating macro-economic vectors."
    }
    method_key = opt.split("-")[0].strip() if "-" in opt else opt.split(":")[0].strip()
    st.divider()
    st.markdown(f"### Methodology: {opt}")
    st.write(methodologies.get(method_key, "Wavelet-based multi-resolution analysis."))
    st.info(f"⚠️ **Risk Policy:** Trailing Stop Loss at {sl_input:.1%}. Recovery requires Z-Score > {rec_sigma}.")
else:
    st.error("Model failure. Check Start Year or data source.")
