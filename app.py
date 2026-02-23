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

# @st.cache_data(ttl=3600, show_spinner=False)  <-- Commented out to allow live debugging
def run_professional_backtest(raw_df, start_yr, model_choice, t_costs_bps, stop_loss_pct, recovery_sigma, _log=None):
    def logger(msg):
        if _log: _log.write(msg)

    # RECTIFICATION: Ensure model_choice is a string to prevent 'not iterable' error
    model_choice = str(model_choice)

    logger("📡 Step 1: Loading raw market data and risk-free rates...")
    
    # Handle tuple if raw_df wasn't unpacked earlier
    if isinstance(raw_df, tuple):
        raw_df = raw_df[0]
    
    if raw_df is None or raw_df.empty:
        logger("❌ Critical Error: DataFrame is empty.")
        return None

    assets = ["TLT", "TBT", "VNQ", "GLD", "SLV", "SPY", "AGG"]
    
    for a in assets:
        if a in raw_df.columns:
            if f"{a}_Ret" not in raw_df.columns:
                raw_df[f"{a}_Ret"] = raw_df[a].pct_change()
            
    try:
        t_cost_pct = float(t_costs_bps) / 10_000
    except (ValueError, TypeError):
        t_cost_pct = 0.0
        
    # Line 68 is now safe because model_choice is guaranteed to be a string
    if "Option F" in model_choice or "Option G" in model_choice:
        logger("🔍 Running Bayesian/Regime logic...")
        # ... rest of logic
        
    from data.processor import build_feature_matrix
    
   
    # --- HMM Training (F & G) ---
    hmm_model = None
    if "Option F" in model_choice or "Option G" in model_choice:
        logger("🧠 Step 2: Training Hidden Markov Model (HMM)...")
        try:
            _, _, idx_ref, _ = build_feature_matrix(raw_df, target_col="TLT")
            m_is_ref = idx_ref.year < start_yr
            hmm_model = RegimeHMM()
            hmm_model.train_and_assign(raw_df.loc[idx_ref[m_is_ref]], assets)
        except Exception as e:
            logger(f"⚠️ HMM Training failed: {e}")

    all_preds = {}
    logger(f"🤖 Step 3: Generating signals using {model_choice}...")
    
    for ticker in assets:
        try:
            logger(f"   -> Processing intelligence stream for {ticker}...")
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker)
            m_oos = idx.year >= start_yr
            oos_len = np.sum(m_oos)
            
            # --- MODEL ROUTING ---
            
            # 1. DEEP LEARNING (I, J, K)
            if "Option I" in model_choice or "Option J" in model_choice or "Option K" in model_choice:
                mode_key = "Option I" if "Option I" in model_choice else ("Option J" if "Option J" in model_choice else "Option K")
                eng = DeepHybridEngine(mode=mode_key)
                fname = {"Option I": "opt_i_cnn.h5", "Option J": "opt_j_cnn_lstm.h5", "Option K": "opt_k_hybrid.h5"}[mode_key]
                eng.load(f"models/{fname}")
                
                oos_indices = np.where(m_oos)[0]
                X_3d = np.array([np.vstack([np.repeat(X[0:1], 20-len(X[max(0, i-19):i+1]), axis=0), X[max(0, i-19):i+1]]) for i in oos_indices])
                
                X_macro = None
                if "Option K" in model_choice:
                    m_df = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].loc[idx[m_oos]].copy()
                    m_df["Mom"], m_df["Vol"], m_df["SPY"] = 0.0, 0.0, 0.0 # Feature padding
                    X_macro = m_df.values
                    
                preds = eng.predict_series(X_3d, X_macro=X_macro)

            # 2. REINFORCEMENT LEARNING (B, C, D)
            elif any(opt in model_choice for opt in ["Option B", "Option C", "Option D"]):
                # A2C / PPO Weights
                eng = A2CEngine() 
                w_file = "models/ppo_weights.pkl" if "Option B" in model_choice else "models/a2c_weights.pkl"
                eng.load(w_file)
                preds = eng.predict_series(X[m_oos])

            # 3. HMM REGIMES (F, G)
            elif "Option F" in model_choice or "Option G" in model_choice:
                macro_oos = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].diff().loc[idx[m_oos]].fillna(0)
                if "Option G" in model_choice:
                    eng = MomentumEngine()
                    eng.load("models/svr_momentum_poly.pkl")
                    base = eng.predict_series(X[m_oos])
                    preds = [base[i] * 1.5 if hmm_model and hmm_model.predict_best_asset(macro_oos.iloc[i:i+1]) == ticker else 0.0 for i in range(oos_len)]
                else:
                    preds = [1.0 if hmm_model and hmm_model.predict_best_asset(macro_oos.iloc[i:i+1]) == ticker else 0.0 for i in range(oos_len)]

            # 4. BAYESIAN / REGIME (E, H)
            elif "Option E" in model_choice or "Option H" in model_choice:
                bf = BayesianFilter()
                conf_data = bf.get_confidence(raw_df[ticker].loc[:idx[m_oos][-1]])
                # Force array conversion and tile if it's a single value to match index length
                conf_vals = np.array(conf_data.values if hasattr(conf_data, 'values') else conf_data).flatten()
                if len(conf_vals) == 1:
                    conf_tail = np.full(oos_len, conf_vals[0])
                else:
                    conf_tail = conf_vals[-oos_len:]
                
                if "Option H" in model_choice:
                    eng = MomentumEngine(); eng.load("models/svr_momentum_poly.pkl")
                    preds = eng.predict_series(X[m_oos]) * conf_tail
                else:
                    preds = conf_tail

            # 5. DEFAULT / OPTION A (SVR)
            else:
                eng = MomentumEngine()
                eng.load("models/svr_momentum_poly.pkl")
                preds = eng.predict_series(X[m_oos])

            all_preds[ticker] = pd.Series(preds, index=idx[m_oos])
            
        except Exception as e:
            logger(f"❌ Error on {ticker}: {e}")
            continue

    logger("📈 Step 4: Running Portfolio Simulation...")
    df_p = pd.DataFrame(all_preds).fillna(0)
    if df_p.empty: return None
    
    common_idx = df_p.index
    equity, current_asset, hwm, in_timeout = 100.0, "CASH", 100.0, False
    rets, hist, confs = [], [], []

    for d in common_idx:
        dp = df_p.loc[d]
        z_score = (dp.max() - dp.mean()) / dp.std() if dp.std() > 0 else 0
        hwm = max(hwm, equity)
        if (equity - hwm) / hwm <= -stop_loss_pct: in_timeout = True 
        if in_timeout and z_score >= recovery_sigma: in_timeout = False
        
        final_sig = "CASH" if in_timeout or dp.max() <= 0 else dp.idxmax()
        
        if final_sig != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = final_sig
            
        day_r = (raw_df.loc[d, "TBILL_3M"]/100)/252 if current_asset == "CASH" else raw_df.loc[d, f"{current_asset}_Ret"]
        equity *= (1 + day_r)
        rets.append(day_r); hist.append(current_asset); confs.append(z_score)

    res = pd.DataFrame(index=common_idx)
    res["Equity"] = (np.array(rets) + 1).cumprod() * 100
    res["Strategy_Ret"] = rets
    res["Drawdown"] = (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    res["SPY"] = (raw_df.loc[common_idx, "SPY_Ret"] + 1).cumprod() * 100
    
    logger("✅ Analysis Complete!")
   # --- RECTIFIED RESULT PACKAGING ---
    res = pd.DataFrame(index=common_idx)
    res["Strategy_Ret"] = rets
    res["Equity"] = (pd.Series(rets) + 1).cumprod().values * 100
    res["Drawdown"] = (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    
    # Use ffill to prevent KeyError/NaN crashes on market holidays (Jan 1st)
    if "SPY" in raw_df.columns:
        res["SPY"] = (raw_df.loc[common_idx, "SPY"].ffill() / raw_df.loc[common_idx, "SPY"].ffill().iloc[0]) * 100
    if "AGG" in raw_df.columns:
        res["AGG"] = (raw_df.loc[common_idx, "AGG"].ffill() / raw_df.loc[common_idx, "AGG"].ffill().iloc[0]) * 100

    # Build the Audit DataFrame with the correct column name 'Return'
    audit_df = pd.DataFrame({
        "Allocation": hist, 
        "Return": rets,       # Mapping 'rets' to 'Return' to satisfy the UI styler
        "Z-Score": confs
    }, index=common_idx)
    
    logger("✅ Analysis Complete!")
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
    
    # 1. Combined Refresh Logic
    if st.button("🔄 Refresh Data & Clear Cache"):
        st.cache_data.clear()
        
        # Trigger sync and update session state immediately
        raw_df_fresh, msg = load_raw_data(force_sync=True) 
        st.session_state['raw_df'] = raw_df_fresh
        
        st.success(msg)
        if raw_df_fresh is not None and not raw_df_fresh.empty:
            st.toast(f"Data Synced: {raw_df_fresh.index[-1].strftime('%Y-%m-%d')}")
        
        st.rerun()

    # 2. Optimized Load Logic (STOP THE FLICKERING)
    if 'raw_df' not in st.session_state:
        raw_df_init, msg = load_raw_data(force_sync=False)
        st.session_state['raw_df'] = raw_df_init

# --- GLOBAL SCOPE RECTIFICATION ---
# This line ensures 'raw_df' is always defined for the backtest engine (Line 273)
raw_df = st.session_state.get('raw_df')
    
    s_yr = st.slider("Backtest Start Year", 2010, 2024, 2015)
    opt = st.radio("Intelligence Engine", [
        "Option A- Wavelet-SVR", 
        "Option B-Wavelet-SVR-PPO", 
        "Option C: Wavelet-A2C", 
        "Option D: Wavelet-SVR-A2C", 
        "Option E: Wavelet-Bayesian-Regime", 
        "Option F: Wavelet-HMM", 
        "Option G- Wavelet-SVR-HMM", 
        "Option H: Wavelet-SVR-Bayesian", 
        "Option I: Wavelet- CNN-LSTM", 
        "Option J: Wavelet-Attention-CNN-LSTM", 
        "Option K- Wavelet- Parallel-Dual-Stream-CNN-LSTM"
    ])
    st.subheader("Risk Gate Controls")
    sl_input = st.slider("Trailing Stop Loss (%)", 8.0, 20.0, 10.0, 0.5) / 100
    rec_sigma = st.slider("Recovery Threshold (Sigma)", 1.0, 2.0, 1.1, 0.1)
    costs = st.number_input("T-Costs (bps)", 0, 50, 10)

# --- DEBUG TOGGLE ---
DEBUG_MODE = True 

# --- UI EXECUTION ---
try:
    if DEBUG_MODE:
        with st.status("🔍 Engine Heartbeat (Debug Mode)", expanded=True) as status:
            st.write("Initializing backtest engine...")
            # Pass raw_df (which we unpacked earlier as a DataFrame)
            out = run_professional_backtest(raw_df, s_yr, opt, costs, sl_input, rec_sigma, _log=status)
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
    else:
        with st.spinner("Processing Model Results..."):
            out = run_professional_backtest(s_yr, opt, costs, sl_input, rec_sigma)

    # 1. Validation: Ensure 'out' exists and contains a valid, non-empty DataFrame
    if out and isinstance(out, dict) and "df" in out and not out["df"].empty:
        df = out["df"]
        st.title("P2 Wavelet Multi-Model")
        
        # --- TOP PREDICTION BANNER ---
        st.markdown(f"""
            <div style="background-color: #f1f8e9; padding: 25px; border-radius: 15px; border: 2px solid #a5d6a7; text-align: center; margin-bottom: 25px;">
                <p style="margin:0; color: #2e7d32; font-size: 14px; font-weight: 700; text-transform: uppercase;">Prediction for NYSE: {get_next_trading_day_simple()}</p>
                <h1 style="margin:5px 0; font-size: 90px; color: #1b5e20; line-height: 1;">{out.get('target', 'CASH')}</h1>
                <p style="margin:0; font-size: 20px; color: #388e3c; font-weight: 500;">Current Z-Score: {float(out.get('conf', 0)):.2f}σ</p>
            </div>
        """, unsafe_allow_html=True)
        
        # --- METRICS GRID ---
        c1, c2, c3, c4, c5 = st.columns(5)
        
        if len(df) > 1 and df['Strategy_Ret'].std() != 0:
            ann_ret = float((df["Equity"].iloc[-1] / 100) ** (252 / len(df)) - 1)
            # --- RECTIFIED METRICS SECTION ---
        # 1. Calculate Daily Risk-Free Rate from TBILL_3M (Annualized % -> Daily Decimal)
        if 'TBILL_3M' in df.columns:
            daily_rf = (df['TBILL_3M'] / 100) / 252
        else:
            daily_rf = 0.0

        # 2. Calculate Sharpe Ratio using the actual benchmark
        excess_ret = df['Strategy_Ret'] - daily_rf
        strat_std = df['Strategy_Ret'].std()

        if strat_std != 0:
            sharpe = float((excess_ret.mean() / strat_std) * np.sqrt(252))
        else:
            sharpe = 0.0

        # 3. Render Metrics
        c1.metric("Annual Return", f"{ann_ret:.2%}")
        c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c3.metric("Max DD (P/T)", f"{float(df['Drawdown'].min()):.2%}")
            
        with c4:
                st.metric("Max DD (Daily)", f"{float(df['Strategy_Ret'].min()):.2%}")
                st.markdown(f'<p class="metric-sub">Worst: <b>{float(df["Strategy_Ret"].min()):.2%}</b> on {df["Strategy_Ret"].idxmin().strftime("%Y-%m-%d")}</p>', unsafe_allow_html=True)
            
        with c5:
                hit_ratio_15d = float((df["Strategy_Ret"].tail(15) > 0).mean())
                st.metric("Hit Ratio (15D)", f"{hit_ratio_15d:.1%}")
                st.markdown(f'<p class="metric-sub">Last 15 Trading Sessions</p>', unsafe_allow_html=True)

       # --- CHARTING ---
        st.subheader("OOS Cumulative Return")
        fig = go.Figure()

        # 1. Strategy Line (Primary)
        if "Equity" in df.columns:
            fig.add_trace(go.Scatter(
            x=df.index, 
            y=df["Equity"], 
            name="P2 Strategy", 
            line=dict(color='#1a73e8', width=3)
            ))

        # 2. Benchmarks (Normalized & Holiday-Safe)
        # We loop through the benchmarks to keep the code clean and prevent KeyErrors
        for bench, color in [("SPY", "#718096"), ("AGG", "#e53e3e")]:
            if bench in df.columns:
            # ffill handles holidays like Jan 1st; dropna finds the first valid trading day for normalization
                bench_series = df[bench].ffill() 
                first_valid = bench_series.dropna().iloc[0] if not bench_series.dropna().empty else None
        
            if first_valid:
            # Normalize to 1.0 to match the Strategy Equity curve
                normalized_bench = bench_series / first_valid
            
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=normalized_bench, 
                name=f"{bench} Bench", 
                line=dict(color=color, dash='dot'),
                connectgaps=True # Bridges the holiday nulls visible in your dataset
            ))

        fig.update_layout(
        template="plotly_white", 
        height=500, 
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Growth of $1.00",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
        st.plotly_chart(fig, use_container_width=True)

       # --- AUDIT TRAIL ---
        if "audit" in out:
            st.subheader("15-Day Audit Trail")
            audit_df = out["audit"].tail(15).copy()
            audit_df.index = audit_df.index.strftime('%Y-%m-%d')

    # Identify the actual return column (likely Strategy_Ret or the ticker_Ret)
    # We rename it to 'Return' for a clean UI and to match your intended styling
        ret_col = next((c for c in audit_df.columns if "Ret" in c), None)
        if ret_col:
                audit_df = audit_df.rename(columns={ret_col: 'Return'})

    # Build the formatting dictionary dynamically based on what exists
    # This prevents the KeyError if Z-Score hasn't been calculated yet
    format_dict = {}
    style_subset = []
    if 'Return' in audit_df.columns:
            format_dict['Return'] = '{:.2%}'
            style_subset.append('Return')
    
    if 'Z-Score' in audit_df.columns:
            format_dict['Z-Score'] = '{:.2f}'
            style_subset.append('Z-Score')

    # Apply styling only to existing columns to avoid "moron" errors/crashes
    try:
        styled_df = audit_df.style.map(
            lambda v: 'color: #d93025' if isinstance(v, (int, float)) and v < 0 else 'color: #188038',
            subset=style_subset
        ).format(format_dict, na_rep="-")
        
        st.dataframe(styled_df, use_container_width=True)
    except Exception:
        # Emergency fallback to raw data if styling fails
        st.dataframe(audit_df, use_container_width=True)

        # --- METHODOLOGY ---
        methodologies = {
            "Option A": "MODWT multi-resolution analysis combined with Polynomial SVR.",
            "Option B": "Hybrid RL-Supervised model using PPO.",
            "Option C": "Advantage Actor-Critic (A2C) optimizing allocation.",
            "Option D": "SVR-A2C Ensemble weighting.",
            "Option E": "Bayesian state-space filtering.",
            "Option F": "Hidden Markov Model (HMM) classification.",
            "Option G": "HMM-Biased SVR.",
            "Option H": "Bayesian-Denoised SVR.",
            "Option I": "CNN-LSTM Deep Learning.",
            "Option J": "Attention-Augmented CNN-LSTM.",
            "Option K": "Parallel Dual-Stream Deep Fusion."
        }
        method_key = opt.split("-")[0].strip() if "-" in opt else opt.split(":")[0].strip()
        st.divider()
        st.markdown(f"### Methodology: {opt}")
        st.write(methodologies.get(method_key, "Wavelet-based multi-resolution analysis."))
        st.info(f"⚠️ **Risk Policy:** Trailing Stop Loss at {sl_input*100:.1f}%. Recovery requires Z-Score > {rec_sigma}.")

    else:
        st.error("Model Engine Error: Backtest returned no data.")
        st.info("Check if your model files are uploaded or if the data range is valid.")

except Exception as e:
    st.error("CRITICAL UI RENDER ERROR")
    st.exception(e)
