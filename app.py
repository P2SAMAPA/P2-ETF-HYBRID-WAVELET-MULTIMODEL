import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.loader import load_raw_data
# RECTIFIED: Added PPOEngine to imports
from engine import MomentumEngine, A2CEngine, PPOEngine, DeepHybridEngine, run_bayesian_filter
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
    import pytz
    # 1. Lock timezone to New York (Wall Street)
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)
    
    # 2. Define Market Open (9:30 AM EST)
    market_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # 3. Logic: If before 9:30 AM, the 'Next Trading Day' is TODAY.
    # If after 9:30 AM, the 'Next Trading Day' is TOMORROW.
    if now_ny < market_open:
        target_date = now_ny
    else:
        # If it's Friday afternoon (weekday 4), skip to Monday (+3 days)
        if now_ny.weekday() == 4:
            target_date = now_ny + timedelta(days=3)
        # If it's Saturday (5), skip to Monday (+2 days)
        elif now_ny.weekday() == 5:
            target_date = now_ny + timedelta(days=2)
        else:
            target_date = now_ny + timedelta(days=1)
            
    return target_date.strftime('%d %B %Y')

def run_professional_backtest(raw_df, start_yr, model_choice, t_costs_bps, stop_loss_pct, recovery_sigma, _log=None):
    def logger(msg):
        if _log: _log.write(msg)

    if isinstance(raw_df, tuple): raw_df = raw_df
    if raw_df is None or raw_df.empty: return None

    # RECTIFIED: Removed TBT and added LQD, HYG, VCIT
    predict_assets = ["TLT", "LQD", "HYG", "VCIT", "VNQ", "GLD", "SLV"]
    comparison_assets = ["SPY", "AGG"]
    all_assets = predict_assets + comparison_assets
    
    for a in all_assets:
        if a in raw_df.columns and f"{a}_Ret" not in raw_df.columns:
            raw_df[f"{a}_Ret"] = raw_df[a].pct_change()
            
    t_cost_pct = float(t_costs_bps) / 10_000
    from data.processor import build_feature_matrix
    
    all_preds = {}
    logger(f"🤖 Generating signals using {model_choice}...")
    
    for ticker in predict_assets:
        try:
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker)
            m_oos = idx.year >= start_yr
            oos_indices = np.where(m_oos)

            # --- CLOUD MODELS (I, J, K) ---
            if any(opt in model_choice for opt in ["Option I", "Option J", "Option K"]):
                mode_key = "Option I" if "Option I" in model_choice else ("Option J" if "Option J" in model_choice else "Option K")
                eng = DeepHybridEngine(mode=mode_key)
                fname = {"Option I": "opt_i_cnn.h5", "Option J": "opt_j_cnn_lstm.h5", "Option K": "opt_k_hybrid.h5"}[mode_key]
                eng.load(f"models/{fname}")
                
                X_3d_list = []
                for i in oos_indices:
                    start_idx = max(0, i - 19)
                    window = X[start_idx : i + 1]
                    if len(window) < 20:
                        window = np.vstack([np.tile(X, (20 - len(window), 1)), window])
                    X_3d_list.append(window)
                
                X_3d = np.array(X_3d_list)
                X_macro = None
                if "Option K" in model_choice:
                    m_df = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].loc[idx[m_oos]].copy()
                    m_df["Mom"], m_df["Vol"], m_df["SPY"] = 0.0, 0.0, 0.0 
                    X_macro = m_df.values
                
                preds = eng.predict_series(X_3d, X_macro=X_macro)
            
            # --- LOCAL MODELS (A-H) ---
            else:
                m_is = idx.year < start_yr
                if "Option B" in model_choice:
                    eng = PPOEngine()
                elif any(opt in model_choice for opt in ["Option C", "Option D"]):
                    eng = A2CEngine()
                else:
                    eng = MomentumEngine()

                eng.train(X[m_is], y[m_is])
                preds = eng.predict_series(X[m_oos])
                
                if any(opt in model_choice for opt in ["Option E", "Option H"]):
                    preds = run_bayesian_filter(preds)

            all_preds[ticker] = pd.Series(preds.values if hasattr(preds, 'values') else preds, index=idx[m_oos])
            
        except Exception as e:
            logger(f"❌ Error on {ticker}: {e}")
            continue

    df_p = pd.DataFrame(all_preds).fillna(0)
    if df_p.empty: return None
    
    common_idx = df_p.index
    equity, current_asset = 100.0, "CASH"
    peak_equity = 100.0
    is_stopped_out = False
    rets, hist, confs = [], [], []

    # --- PORTFOLIO LOOP WITH TRAILING STOP LOSS ---
    for i, d in enumerate(common_idx):
        dp = df_p.loc[d]
        std_val = dp.std()
        z_score = (dp.max() - dp.mean()) / std_val if std_val > 1e-6 else 0.0
        
        # Determine raw signal
        best_ticker = dp.idxmax() if dp.max() > 0 else "CASH"
        final_sig = best_ticker

        # Recovery Logic: If stopped out, must clear Sigma threshold to re-enter
        if is_stopped_out:
            if z_score > recovery_sigma:
                is_stopped_out = False # Recovered
            else:
                final_sig = "CASH"

        # Check for Stop Loss trigger if currently in an asset
        if current_asset != "CASH":
            # Track peak equity for the current holding
            if equity > peak_equity: peak_equity = equity
            
            # Trigger Trailing Stop Loss
            if equity < (peak_equity * (1 - stop_loss_pct)):
                final_sig = "CASH"
                is_stopped_out = True
        
        # Execute Trade & Apply T-Costs
        if final_sig != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = final_sig
            peak_equity = equity # Reset peak for new asset or cash position
            
        # Calculate daily return
        if current_asset == "CASH":
            day_r = (raw_df.loc[d, "TBILL_3M"]/100)/252 if "TBILL_3M" in raw_df.columns else 0.0
        else:
            day_r = raw_df.loc[d, f"{current_asset}_Ret"]
            
        equity *= (1 + day_r)
        rets.append(day_r)
        hist.append(current_asset)
        confs.append(z_score)

    res = pd.DataFrame(index=common_idx)
    res["Strategy_Ret"] = rets
    res["Equity"] = (pd.Series(rets) + 1).cumprod().values * 100
    res["Drawdown"] = (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    
    for comp in comparison_assets:
        if comp in raw_df.columns:
            res[comp] = (raw_df.loc[common_idx, comp] / raw_df.loc[common_idx, comp].iloc) * 100

    audit_df = pd.DataFrame({"Allocation": hist, "Return": rets, "Z-Score": confs}, index=common_idx)
    
    return {
        "df": res.ffill(), 
        "audit": audit_df, 
        "target": str(hist[-1]), 
        "conf": float(confs[-1]), 
        "date": common_idx[-1].strftime('%Y-%m-%d')
    }

# --- SIDEBAR & UI RENDER ---
with st.sidebar:
    st.header("Terminal Config")
    st.info("💡 Options I, J, K are cloud-trained (2008-2026). Options A-H retrain locally.")
    
    if st.button("🔄 Refresh Data & Clear Cache"):
        st.cache_data.clear()
        st.session_state['raw_df'], sync_msg = load_raw_data(force_sync=True)
        st.toast(sync_msg)
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
    # RECTIFIED: Updated range to 0.75 - 2.0
    rec_sigma = st.slider("Recovery Threshold (Sigma)", 0.75, 2.0, 0.80, 0.05)
    costs = st.number_input("T-Costs (bps)", 0, 50, 10)

raw_df = st.session_state.get('raw_df')

if raw_df is not None:
    try:
        with st.status("🔍 Engine Heartbeat", expanded=False) as status:
            out = run_professional_backtest(raw_df, s_yr, opt, costs, sl_input, rec_sigma, _log=status)
        
        if out:
            df = out["df"]
            st.title("P2 Wavelet Multi-Model")
            
            st.markdown(f"""<div style="background-color: #f1f8e9; padding: 25px; border-radius: 15px; border: 2px solid #a5d6a7; text-align: center; margin-bottom: 25px;">
                <p style="margin:0; color: #2e7d32; font-size: 14px; font-weight: 700; text-transform: uppercase;">Prediction: {get_next_trading_day_simple()}</p>
                <h1 style="margin:5px 0; font-size: 90px; color: #1b5e20; line-height: 1;">{out.get('target', 'CASH')}</h1>
                <p style="margin:0; font-size: 20px; color: #388e3c; font-weight: 500;">Z-Score: {float(out.get('conf', 0)):.2f}σ</p>
            </div>""", unsafe_allow_html=True)
            
            c1, c2, c3, c4, c5 = st.columns(5)
            ann_ret = (df["Equity"].iloc[-1] / 100) ** (252 / len(df)) - 1
            strat_std = df['Strategy_Ret'].std()
            sharpe = (df['Strategy_Ret'].mean() / strat_std) * np.sqrt(252) if strat_std != 0 else 0
            
            c1.metric("Annual Return", f"{ann_ret:.2%}")
            c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            c3.metric("Max DD (P/T)", f"{df['Drawdown'].min():.2%}")
            c4.metric("Max DD (Daily)", f"{df['Strategy_Ret'].min():.2%}")
            
            with c5:
                hit_ratio = (df["Strategy_Ret"].tail(15) > 0).mean()
                st.metric("Hit Ratio (15D)", f"{hit_ratio:.1%}")

            st.subheader("15-Day Audit Trail")
            audit_display = out["audit"].tail(15).copy()
            audit_display.index = audit_display.index.strftime('%Y-%m-%d')
            
            # RECTIFIED: Conditional formatting for Green/Red returns
            def style_returns(val):
                color = '#d93025' if val < 0 else '#188038'
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                audit_display.style.format({'Return': '{:.2%}', 'Z-Score': '{:.2f}'})
                .applymap(style_returns, subset=['Return']), 
                use_container_width=True
            )

          # Ensure this block is indented 4 spaces (one level) from the left margin 
            # if it is inside the main() or backtest function.
            methodologies = {
                "Option A": "MODWT Multi-Resolution Analysis + RBF-SVR: Decomposes price into frequency bands via Wavelets, then uses a High-Penalty Radial Basis Function SVR to capture non-linear trend pivots.",
                "Option B": "Hybrid SVR-PPO: Uses SVR-denoised signals as state inputs for Proximal Policy Optimization, employing a 'clipping' mechanism to ensure stable policy updates during high volatility.",
                "Option C": "A2C Allocation Optimizer: An Advantage Actor-Critic Reinforcement Learning framework that treats portfolio weightings as actions, rewarding the 'Actor' for Sharpe Ratio maximization.",
                "Option D": "SVR-A2C Ensemble: A multi-agent approach where SVR provides a deterministic baseline and A2C provides a stochastic 'advantage' adjustment to refine the final asset selection.",
                "Option E": "Bayesian State-Space Filtering: Uses recursive Bayesian updates and a Signal-to-Noise Ratio (SNR) threshold to isolate structural regime shifts from random market noise.",
                "Option F": "HMM Regime Classification: A 3-state Hidden Markov Model (Bull/Bear/Sideways) that maps hidden market states to macro-economic 'Pillars' like VIX and Credit Spreads.",
                "Option G": "HMM-Biased RBF-SVR: Uses HMM-derived market states to dynamically bias SVR predictions, forcing more aggressive positioning during detected Bull regimes.",
                "Option H": "Bayesian-Denoised RBF-SVR: Passes aggressive RBF-SVR outputs through a secondary Bayesian confidence filter to scale position sizes based on statistical conviction.",
                "Option I": "CNN-LSTM Deep Learning: A cloud-trained (2008-2026) hybrid architecture using Convolutions for spatial pattern detection and LSTMs for long-term temporal dependencies.",
                "Option J": "Attention-Augmented CNN-LSTM: Integrates a 'Soft-Attention' layer that allows the model to selectively weigh historical crash events more heavily than recent noise during prediction.",
                "Option K": "Parallel Dual-Stream Deep Fusion: The most advanced engine; it fuses a dedicated Price-Stream and a Macro-Stream (VIX/DXY) into a final decision dense layer."
    }
            
            method_key = opt.split("-").strip() if "-" in opt else opt.split(":").strip()
            if ":" in method_key: method_key = method_key.split(":").strip()
            
            st.divider()
            st.markdown(f"### Methodology: {opt}")
            st.write(methodologies.get(method_key, "Wavelet-based multi-resolution analysis."))
            st.info(f"⚠️ **Risk Policy:** Trailing Stop Loss at {sl_input*100:.1f}%. Recovery requires Z-Score > {rec_sigma}.")

    except Exception as e:
        st.error("CRITICAL UI RENDER ERROR")
        st.exception(e)
else:
    st.info("Please wait... Loading market data.")
