import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.loader import load_raw_data
from engine import MomentumEngine, A2CEngine, DeepHybridEngine 

st.set_page_config(page_title="P2 ETF WAVELET SVR MULTI MODEL", layout="wide", initial_sidebar_state="collapsed")

# --- UI STYLING ---
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
    h1, h2, h3 { color: #202124 !important; font-family: 'Segoe UI', sans-serif; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def run_professional_backtest(start_yr, model_choice, t_costs_bps):
    raw_df = load_raw_data()
    assets = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
    t_cost_pct = t_costs_bps / 10_000
    
    from data.processor import build_feature_matrix
    from analytics.regime import RegimeHMM
    
    all_preds = {} 
    oos_mask_global = raw_df.index.year >= start_yr
    is_mask_global = raw_df.index.year < start_yr

    # --- REGIME ENGINES ---
    hmm_signals = {}
    if any(x in model_choice for x in ["Option D: Wavelet- HMM", "Option E", "Option G", "Option H"]):
        hmm_engine = RegimeHMM(n_states=3)
        hmm_engine.train_and_assign(raw_df[is_mask_global], assets)
        macro_oos = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]][oos_mask_global].diff().fillna(0)
        for date, row in macro_oos.iterrows():
            hmm_signals[date] = hmm_engine.predict_best_asset(row.values.reshape(1, -1))

    # --- MODEL PREDICTIONS ---
    for ticker in assets:
        try:
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker)
            oos_mask = idx.year >= start_yr
            is_mask = idx.year < start_yr
            
            if any(opt in model_choice for opt in ["Option I", "Option J", "Option K"]):
                active_mode = "Option K" if "Option K" in model_choice else "Option J" if "Option J" in model_choice else "Option I"
                engine = DeepHybridEngine(mode=active_mode)
                lookback = 20
                full_X_values = X.values
                oos_indices = np.where(oos_mask)[0]
                X_3d_list = []
                for i in oos_indices:
                    window = full_X_values[max(0, i-lookback+1):i+1]
                    if len(window) < lookback:
                        window = np.vstack([np.repeat(window[0:1], lookback - len(window), axis=0), window])
                    X_3d_list.append(window)
                
                X_3d = np.array(X_3d_list)
                X_macro = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].loc[idx[oos_mask]].values if "Option K" in model_choice else None
                preds = engine.predict_series(X_3d, X_macro=X_macro)
            elif any(opt in model_choice for opt in ["Option C", "Option D: Wavelet-SVR-A2C"]):
                engine = A2CEngine(); engine.train(X[is_mask], y[is_mask])
                preds = engine.predict_series(X[oos_mask])
            else:
                engine = MomentumEngine(c_param=700.0); engine.train(X[is_mask], y[is_mask])
                preds = engine.predict_series(X[oos_mask])
            
            all_preds[ticker] = pd.Series(preds, index=idx[oos_mask])
        except: continue

    pred_df = pd.DataFrame(all_preds).dropna() if all_preds else pd.DataFrame()
    threshold = 0.0015 if "Option B" in model_choice else 0.0

    # --- EXECUTION LOOP ---
    oos_idx = raw_df.index[oos_mask_global]
    equity, current_asset = 100.0, "CASH"
    strat_rets, asset_history, conf_history = [], [], []
    hwm, loss_streak, in_timeout = 100.0, 0, False

    for date in oos_idx:
        hwm = max(hwm, equity)
        drawdown = (equity - hwm) / hwm
        if drawdown <= -0.12: loss_streak += 1
        else: loss_streak = 0
        if loss_streak >= 2: in_timeout = True

        daily_preds = pred_df.loc[date] if (not pred_df.empty and date in pred_df.index) else None
        raw_signal = daily_preds.idxmax() if (daily_preds is not None and daily_preds.max() > threshold) else "CASH"
        conf = 0.5 + (0.18 * ((daily_preds.max() - daily_preds.mean()) / daily_preds.std())) if (daily_preds is not None and daily_preds.std() > 0) else 0.5

        if in_timeout:
            if conf >= 0.75: in_timeout = False; final_signal = raw_signal
            else: final_signal = "CASH"
        else:
            final_signal = raw_signal

        if final_signal != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = final_signal
        
        day_ret = (raw_df.loc[date, "TBILL_3M"]/100)/252 if current_asset == "CASH" else raw_df.loc[date, f"{current_asset}_Ret"]
        equity *= (1 + day_ret)
        strat_rets.append(day_ret)
        asset_history.append(current_asset)
        conf_history.append(conf)

    res = pd.DataFrame(index=oos_idx)
    res["Strategy_Ret"] = strat_rets
    res["Equity"] = (pd.Series(strat_rets, index=oos_idx).add(1).cumprod() * 100)
    res["Drawdown"] = (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    res["RF"] = (raw_df.loc[oos_idx, "TBILL_3M"] / 100) / 252
    for b in ["SPY", "AGG"]: res[b] = (raw_df.loc[oos_idx, f"{b}_Ret"].add(1).cumprod() * 100)

    return {"df": res, "audit": pd.DataFrame({"Allocation": asset_history, "Daily_Return": strat_rets}, index=oos_idx), "target": asset_history[-1], "confidence": conf_history[-1], "last_date": oos_idx[-1].strftime('%Y-%m-%d')}

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("Terminal Config")
    s_yr = st.slider("Backtest Start Year", 2010, 2024, 2015)
    opt = st.radio("Model Logic", [
        "Option A- Wavelet-SVR", "Option B-Wavelet-SVR-PPO", "Option C: Wavelet-A2C", 
        "Option D: Wavelet-SVR-A2C", "Option D: Wavelet- HMM", "Option E: Wavelet-Bayesian-Regime", 
        "Option G- Wavelet-SVR-HMM", "Option H: Wavelet-SVR-Bayesian", "Option I: Wavelet- CNN-LSTM", 
        "Option J: Wavelet-Attention-CNN-LSTM", "Option K- Wavelet- Parallel-Dual-Stream-CNN-LSTM"
    ])
    costs = st.number_input("T-Costs (bps)", 0, 50, 10)

output = run_professional_backtest(s_yr, opt, costs)

# --- UI HEADER & BANNER ---
st.markdown("<h1 style='text-align: center; color: #1a73e8;'>🦅 P2 ETF WAVELET SVR MULTI MODEL</h1>", unsafe_allow_html=True)
st.success(f"✅ Data updated till {output['last_date']}")

st.markdown(f"""<div style="background-color: #f1f8e9; padding: 20px; border-radius: 12px; border: 2px solid #a5d6a7; text-align: center;">
    <p style="margin:0; color: #2e7d32; font-size: 14px; font-weight: 700;">TARGET ALLOCATION</p>
    <h1 style="margin:5px 0; font-size: 80px; color: #1b5e20;">{output['target']}</h1>
    <p style="margin:0; font-size: 14px; color: #2e7d32; font-weight: bold;">SIGNAL CONFIDENCE: {output['confidence']:.0%}</p>
</div>""", unsafe_allow_html=True)

# --- METRICS ---
df = output["df"]
ann_ret = (df["Equity"].iloc[-1] / 100) ** (252 / len(df)) - 1
ann_vol = df["Strategy_Ret"].std() * np.sqrt(252)
kelly_calc = 0.5 * (ann_ret / (ann_vol**2)) if ann_vol > 0 and ann_ret > 0 else 0
kelly_display = f"{min(kelly_calc, 0.50):.1%}" if output['target'] != "CASH" else "0.0%"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ann. Return", f"{ann_ret:.2%}")
c2.metric("Sharpe Ratio", f"{((df['Strategy_Ret']-df['RF']).mean()/df['Strategy_Ret'].std())*np.sqrt(252):.2f}")
c3.metric("Max DD (Peak to Trough)", f"{df['Drawdown'].min():.2%}")
c4.metric("Hit Ratio(15Days)", f"{(pd.Series(df['Strategy_Ret']).tail(15) > 0).mean():.0%}")
c5.metric("1/2 Kelly Factor", kelly_display)

# --- CHART (Normalized) ---
st.subheader("Cumulative Return Chart (Normalized to 100)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Equity"], name="Strategy", line=dict(color='#1a73e8', width=3)))
fig.add_trace(go.Scatter(x=df.index, y=df["SPY"], name="SPY (Normalized)", line=dict(color='rgba(128,128,128,0.6)', dash='dash')))
fig.add_trace(go.Scatter(x=df.index, y=df["AGG"], name="AGG (Normalized)", line=dict(color='rgba(165,42,42,0.6)', dash='dot')))
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# --- AUDIT ---
st.subheader("15-Day Audit Trail")
audit = output["audit"].tail(15).copy()
audit.index = audit.index.strftime('%Y-%m-%d')
st.dataframe(
    audit.style.applymap(lambda x: 'color: #155724; background-color: #d4edda' if isinstance(x, float) and x > 0 else 'color: #721c24; background-color: #f8d7da' if isinstance(x, float) and x < 0 else '', subset=['Daily_Return']).format({'Daily_Return': '{:.2%}'}),
    use_container_width=True
)

# --- METHODOLOGY ---
methodologies = {
    "Option A": "MODWT Wavelet denoising + Polynomial SVR for momentum capture.",
    "Option B": "Wavelet-SVR with PPO-agent adaptive thresholding.",
    "Option C": "Advantage Actor-Critic (RL) with wavelet-denoised features.",
    "Option D: Wavelet-SVR-A2C": "Hybrid SVR-RL ensemble on wavelet coefficients.",
    "Option D: Wavelet- HMM": "Hidden Markov Model regime detection on wavelet scales.",
    "Option E": "Bayesian state-space filtering for state transition probability.",
    "Option G": "SVR combined with macro-HMM regime biasing.",
    "Option H": "SVR with Bayesian filtering for noise reduction.",
    "Option I": "3D CNN-LSTM architecture on wavelet tensor inputs.",
    "Option J": "CNN-LSTM enhanced with spatial-temporal Attention mechanisms.",
    "Option K": "Parallel Dual-Stream Network fusing price sequences with macro vectors."
}
selected_method = methodologies.get(opt.split("-")[0].strip(), methodologies.get(opt.split(":")[0].strip(), "Wavelet multi-model approach."))

st.markdown(f"### Methodology: {opt}")
st.write(selected_method)
st.info("⚠️ **Risk Gate:** Drops of **12% from HWM** (2-day confirmation) trigger CASH. Re-entry requires **Signal Confidence > 75%**.")
