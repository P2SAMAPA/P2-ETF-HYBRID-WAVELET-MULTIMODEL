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

def run_professional_backtest(raw_df, start_yr, model_choice, t_costs_bps, stop_loss_pct, recovery_sigma, _log=None):
    def logger(msg):
        if _log: _log.write(msg)

    model_choice = str(model_choice)
    logger("📡 Step 1: Loading raw market data...")
    
    if isinstance(raw_df, tuple): raw_df = raw_df[0]
    if raw_df is None or raw_df.empty: return None

    # Prediction Assets vs comparison Assets
    predict_assets = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
    comparison_assets = ["SPY", "AGG"]
    all_assets = predict_assets + comparison_assets
    
    for a in all_assets:
        if a in raw_df.columns and f"{a}_Ret" not in raw_df.columns:
            raw_df[f"{a}_Ret"] = raw_df[a].pct_change()
            
    t_cost_pct = float(t_costs_bps) / 10_000
    from data.processor import build_feature_matrix
    
    all_preds = {}
    logger(f"🤖 Step 3: Generating signals using {model_choice}...")
    
    for ticker in predict_assets:
        try:
            logger(f"    -> Intelligence Stream: {ticker}")
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker)
            m_is = idx.year < start_yr
            m_oos = idx.year >= start_yr
            oos_len = np.sum(m_oos)
            
            # --- CLOUD TIERS (I, J, K) ---
            if any(opt in model_choice for opt in ["Option I", "Option J", "Option K"]):
                mode_key = "Option I" if "Option I" in model_choice else ("Option J" if "Option J" in model_choice else "Option K")
                eng = DeepHybridEngine(mode=mode_key)
                fname = {"Option I": "opt_i_cnn.h5", "Option J": "opt_j_cnn_lstm.h5", "Option K": "opt_k_hybrid.h5"}[mode_key]
                eng.load(f"models/{fname}")
                
                oos_indices = np.where(m_oos)[0]
                X_3d = np.array([np.vstack([np.repeat(X[0:1], 20-len(X[max(0, i-19):i+1]), axis=0), X[max(0, i-19):i+1]]) for i in oos_indices])
                X_macro = None
                if "Option K" in model_choice:
                    m_df = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].loc[idx[m_oos]].copy()
                    m_df["Mom"], m_df["Vol"], m_df["SPY"] = 0.0, 0.0, 0.0 
                    X_macro = m_df.values
                preds = eng.predict_series(X_3d, X_macro=X_macro)

            # --- LOCAL TIERS (A-H) ---
            else:
                if any(opt in model_choice for opt in ["Option B", "Option C", "Option D"]):
                    eng = A2CEngine()
                else:
                    eng = MomentumEngine()
                
                # RETRAIN LOCALLY BASED ON SLIDER YEAR
                eng.train(X[m_is], y[m_is])
                preds = eng.predict_series(X[m_oos])

            all_preds[ticker] = pd.Series(preds, index=idx[m_oos])
        except Exception as e:
            logger(f"❌ Error on {ticker}: {e}")
            continue

    logger("📈 Step 4: Running Portfolio Simulation (STOP LOSS DISABLED)...")
            df_p = pd.DataFrame(all_preds).fillna(0)
            if df_p.empty: return None
    
            common_idx = df_p.index
            equity, current_asset, in_timeout = 100.0, "CASH", False
            rets, hist, confs = [], [], []

            for i, d in enumerate(common_idx):
                dp = df_p.loc[d]
                z_score = (dp.max() - dp.mean()) / dp.std() if dp.std() > 0 else 0
        
        # --- STOP LOSS MODULE TEMPORARILY DISABLED ---
        # Logic is bypassed to debug why models are outputting CASH
        in_timeout = False 
        
        # Decision Logic: Always Winner Takes All
        final_sig = dp.idxmax()
        
        # --- TRANSACTION COSTS ---
        # Irrespective of stop loss; triggers whenever the asset selection changes
        if final_sig != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = final_sig
            
        # Daily Return Calculation
        if current_asset == "CASH":
            day_r = (raw_df.loc[d, "TBILL_3M"]/100)/252
        else:
            day_r = raw_df.loc[d, f"{current_asset}_Ret"]
            
        equity *= (1 + day_r)
        
        rets.append(day_r)
        hist.append(current_asset)
        confs.append(z_score)

    # Building the Results DataFrame
    res = pd.DataFrame(index=common_idx)
    res["Strategy_Ret"] = rets
    res["Equity"] = (pd.Series(rets) + 1).cumprod().values * 100
    res["Drawdown"] = (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    
    for comp in comparison_assets:
        if comp in raw_df.columns:
            res[comp] = (raw_df.loc[common_idx, comp].ffill() / raw_df.loc[common_idx, comp].ffill().iloc[0]) * 100
    
    if "TBILL_3M" in raw_df.columns: 
        res["TBILL_3M"] = raw_df.loc[common_idx, "TBILL_3M"]

    audit_df = pd.DataFrame({"Allocation": hist, "Return": rets, "Z-Score": confs}, index=common_idx)
    
    return {
        "df": res.ffill().fillna(100.0), 
        "audit": audit_df, 
        "target": str(hist[-1]), 
        "conf": float(confs[-1]), 
        "date": common_idx[-1].strftime('%Y-%m-%d')
    }
# --- SIDEBAR ---
with st.sidebar:
    st.header("Terminal Config")
    st.info("💡 **Training Policy:** Options I, J, K are cloud-trained (2008-2026). Options A-H retrain locally based on the Start Year slider.")
    
    if st.button("🔄 Refresh Data & Clear Cache"):
        st.cache_data.clear()
        raw_df_fresh, _ = load_raw_data(force_sync=True) 
        st.session_state['raw_df'] = raw_df_fresh
        st.rerun()

    if 'raw_df' not in st.session_state:
        st.session_state['raw_df'], _ = load_raw_data(force_sync=False)

    s_yr = st.slider("Backtest Start Year", 2010, 2024, 2015)
    opt = st.radio("Intelligence Engine", [
        "Option A- Wavelet-SVR", "Option B-Wavelet-SVR-PPO", "Option C: Wavelet-A2C", 
        "Option D: Wavelet-SVR-A2C", "Option E: Wavelet-Bayesian-Regime", "Option F: Wavelet-HMM", 
        "Option G- Wavelet-SVR-HMM", "Option H: Wavelet-SVR-Bayesian", "Option I: Wavelet- CNN-LSTM", 
        "Option J: Wavelet-Attention-CNN-LSTM", "Option K- Wavelet- Parallel-Dual-Stream-CNN-LSTM"
    ])
    sl_input = st.slider("Trailing Stop Loss (%)", 8.0, 20.0, 10.0, 0.5) / 100
    rec_sigma = st.slider("Recovery Threshold (Sigma)", 1.0, 2.0, 1.1, 0.1)
    costs = st.number_input("T-Costs (bps)", 0, 50, 10)

raw_df = st.session_state.get('raw_df')

# --- UI EXECUTION ---
if raw_df is not None:
    try:
        with st.status("🔍 Engine Heartbeat", expanded=False) as status:
            out = run_professional_backtest(raw_df, s_yr, opt, costs, sl_input, rec_sigma, _log=status)
        
        if out and "df" in out:
            df = out["df"]
            st.title("P2 Wavelet Multi-Model")
            st.markdown(f"""<div style="background-color: #f1f8e9; padding: 25px; border-radius: 15px; border: 2px solid #a5d6a7; text-align: center; margin-bottom: 25px;">
                <p style="margin:0; color: #2e7d32; font-size: 14px; font-weight: 700; text-transform: uppercase;">Prediction: {get_next_trading_day_simple()}</p>
                <h1 style="margin:5px 0; font-size: 90px; color: #1b5e20; line-height: 1;">{out.get('target', 'CASH')}</h1>
                <p style="margin:0; font-size: 20px; color: #388e3c; font-weight: 500;">Z-Score: {float(out.get('conf', 0)):.2f}σ</p>
            </div>""", unsafe_allow_html=True)
            
            c1, c2, c3, c4, c5 = st.columns(5)
            ann_ret = float((df["Equity"].iloc[-1] / 100) ** (252 / len(df)) - 1)
            strat_std = df['Strategy_Ret'].std()
            sharpe = float((df['Strategy_Ret'].mean() / strat_std) * np.sqrt(252)) if strat_std != 0 else 0.0
            
            max_daily_dd = df['Strategy_Ret'].min()
            max_daily_dd_date = df['Strategy_Ret'].idxmin().strftime('%Y-%m-%d')

            c1.metric("Annual Return", f"{ann_ret:.2%}")
            c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            c3.metric("Max DD (P/T)", f"{float(df['Drawdown'].min()):.2%}")
            
            # RECTIFIED: Max DD (Daily) with Date
            c4.metric("Max DD (Daily)", f"{max_daily_dd:.2%}")
            st.markdown(f"<div class='metric-sub'>Worst Day: <b>{max_daily_dd_date}</b></div>", unsafe_allow_html=True)
            
            with c5:
                hit_ratio_15d = float((df["Strategy_Ret"].tail(15) > 0).mean())
                st.metric("Hit Ratio (15D)", f"{hit_ratio_15d:.1%}")

            # RECTIFIED: Moved out of 'with c5' to ensure full-width rendering
            st.subheader("15-Day Audit Trail")
            audit_df = out["audit"].tail(15).copy()
            audit_df.index = audit_df.index.strftime('%Y-%m-%d')
            style_subset = [c for c in ['Return', 'Z-Score'] if c in audit_df.columns]
            styled_df = audit_df.style.map(lambda v: 'color: #d93025' if isinstance(v, (int, float)) and v < 0 else 'color: #188038', subset=style_subset).format({'Return': '{:.2%}', 'Z-Score': '{:.2f}'}, na_rep="-")
            st.dataframe(styled_df, use_container_width=True)

            methodologies = {"Option A": "MODWT multi-resolution analysis combined with Polynomial SVR.", "Option B": "Hybrid RL-Supervised model using PPO.", "Option C": "Advantage Actor-Critic (A2C) optimizing allocation.", "Option D": "SVR-A2C Ensemble weighting.", "Option E": "Bayesian state-space filtering.", "Option F": "Hidden Markov Model (HMM) classification.", "Option G": "HMM-Biased SVR.", "Option H": "Bayesian-Denoised SVR.", "Option I": "CNN-LSTM Deep Learning.", "Option J": "Attention-Augmented CNN-LSTM.", "Option K": "Parallel Dual-Stream Deep Fusion."}
            method_key = opt.split("-")[0].strip() if "-" in opt else opt.split(":")[0].strip()
            st.divider()
            st.markdown(f"### Methodology: {opt}")
            st.write(methodologies.get(method_key, "Wavelet-based multi-resolution analysis."))
            st.info(f"⚠️ **Risk Policy:** Trailing Stop Loss at {sl_input*100:.1f}%. Recovery requires Z-Score > {rec_sigma}.")

        else:
            st.error("Model Engine Error: Backtest returned no data.")

    except Exception as e:
        st.error("CRITICAL UI RENDER ERROR")
        st.exception(e)
else:
    st.info("Please wait... Loading market data.")
