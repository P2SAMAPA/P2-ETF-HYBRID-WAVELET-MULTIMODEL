import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.loader import load_raw_data
# RECTIFIED IMPORT: Point to the root engine.py
from engine import MomentumEngine, A2CEngine, DeepHybridEngine 

st.set_page_config(page_title="P2 ETF WAVELET SVR MULTI MODEL", layout="wide", initial_sidebar_state="collapsed")

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = "Never (Initial Load)"

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

    hmm_signals = {}
    if "HMM" in model_choice or "Option G" in model_choice:
        hmm_engine = RegimeHMM(n_states=3)
        hmm_engine.train_and_assign(raw_df[is_mask_global], assets)
        macro_cols = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]
        macro_oos = raw_df[macro_cols][oos_mask_global].diff().fillna(0)
        for date, row in macro_oos.iterrows():
            hmm_signals[date] = hmm_engine.predict_best_asset(row.values.reshape(1, -1))

    bsts_filter = BayesianFilter() if "BSTS" in model_choice else None

    if not (model_choice.startswith("Option E") or model_choice.startswith("Option G")):
        for ticker in assets:
            try:
                X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker)
                is_mask = idx.year < start_yr
                oos_mask = idx.year >= start_yr
                
                # RECTIFICATION START: Handle 3D Input for Deep Learning
                if any(opt in model_choice for opt in ["Option I", "Option J", "Option K"]):
                    active_mode = "Option K" if "Option K" in model_choice else \
                                  "Option J" if "Option J" in model_choice else "Option I"
                    
                    engine = DeepHybridEngine(mode=active_mode)
                    
                    # Convert 2D feature matrix to 3D sliding window for LSTM/CNN
                    lookback = 20
                    full_X_values = X.values if hasattr(X, 'values') else X
                    oos_indices = np.where(oos_mask)[0]
                    
                    X_3d_list = []
                    for i in oos_indices:
                        window = full_X_values[max(0, i-lookback+1):i+1]
                        if len(window) < lookback:
                            # Padding if history is insufficient
                            pad_size = lookback - len(window)
                            window = np.vstack([np.repeat(window[0:1], pad_size, axis=0), window])
                        X_3d_list.append(window)
                    
                    X_3d = np.array(X_3d_list)
                    
                    X_macro = None
                    if active_mode == "Option K":
                        macro_cols = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]
                        X_macro = raw_df[macro_cols].loc[idx[oos_mask]].values
                    
                    preds = engine.predict_series(X_3d, X_macro=X_macro)
                # RECTIFICATION END
                
                elif "Option C" in model_choice:
                    engine = A2CEngine()
                    engine.train(X[is_mask], y[is_mask])
                    preds = engine.predict_series(X[oos_mask])
                
                else:
                    engine = MomentumEngine(c_param=700.0)
                    engine.train(X[is_mask], y[is_mask])
                    preds = engine.predict_series(X[oos_mask])
                
                all_preds[ticker] = pd.Series(preds, index=idx[oos_mask])
            except:
                continue

    pred_df = pd.DataFrame(all_preds).dropna() if all_preds else pd.DataFrame()

    # Threshold Logic
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
        if model_choice.startswith("Option E") or model_choice.startswith("Option G"):
            signal_asset = hmm_signals.get(date, "CASH")
        elif model_choice.startswith("Option F"):
            daily_preds = pred_df.loc[date] if (not pred_df.empty and date in pred_df.index) else None
            hmm_gate = hmm_signals.get(date, "CASH")
            signal_asset = daily_preds.idxmax() if (daily_preds is not None and daily_preds.max() > threshold and hmm_gate != "CASH") else "CASH"
        elif model_choice.startswith("Option H"):
            daily_preds = pred_df.loc[date] if (not pred_df.empty and date in pred_df.index) else None
            if daily_preds is not None and daily_preds.max() > threshold:
                conf = bsts_filter.get_confidence(raw_df.loc[:date, daily_preds.idxmax()])
                signal_asset = daily_preds.idxmax() if conf > 0.65 else "CASH"
            else:
                signal_asset = "CASH"
        else:
            if not pred_df.empty and date in pred_df.index:
                daily_preds = pred_df.loc[date]
                signal_asset = daily_preds.idxmax() if daily_preds.max() > threshold else "CASH"
            else:
                signal_asset = "CASH"

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

    res = pd.DataFrame(index=oos_idx)
    res["Strategy_Ret"] = strat_rets
    res["Equity"] = (pd.Series(strat_rets, index=oos_idx).add(1).cumprod() * 100)
    res["Drawdown"] = (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    res["RF"] = (raw_df.loc[oos_idx, "TBILL_3M"] / 100) / 252
    for b in ["SPY", "AGG"]: res[b] = (raw_df.loc[oos_idx, f"{b}_Ret"].add(1).cumprod() * 100)

    confidence_val = 0.50 
    if not pred_df.empty:
        last_preds = pred_df.iloc[-1]
        std = last_preds.std()
        z_score = (last_preds.max() - last_preds.mean()) / std if std > 1e-6 else 0
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
        "strat_rets_raw": strat_rets,
        "is_ghost": pred_df.abs().sum().sum() == 0 if not pred_df.empty else True
    }

st.markdown("<h1 style='text-align: center; color: #1a73e8; margin-bottom: 0;'>🦅 P2 ETF WAVELET SVR MULTI MODEL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5f6368; font-weight: 500;'>Institutional Strategy Performance & Signal Console</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Terminal Config")
    if st.button("🔄 Force Data Refresh"):
        st.cache_data.clear()
        st.rerun()

    s_yr = st.slider("Backtest Start Year", 2010, 2024, 2015)
    opt = st.radio("Model Logic", [
        "Option A - Wavelet-SVR", "Option B - Wavelet-SVR-PPO", "Option C - Wavelet-A2C", 
        "Option D - Wavelet-SVR-A2C", "Option E - Wavelet-HMM", "Option F - Wavelet-SVR-HMM", 
        "Option G - Wavelet-BSTS", "Option H - Wavelet-SVR-BSTS",
        "Option I - Wavelet-CNN-LSTM", 
        "Option J - Wavelet-SVR-CNN-LSTM", 
        "Option K - Parallel-Dual-Stream"
    ])
    costs = st.number_input("T-Costs (bps)", 0, 50, 10)

output = run_professional_backtest(s_yr, opt, costs)

if output:
    # Diagnostic Warning for Options I, J, K
    if output["is_ghost"] and any(x in opt for x in ["Option I", "Option J", "Option K"]):
        st.warning("⚠️ ENGINE DATA MISSING: Model files (.h5) could not be loaded. Showing Risk-Free baseline (CASH).")

    data = output["df"]
    strat_rets = output["strat_rets_raw"]
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

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    excess = data["Strategy_Ret"] - data["RF"]
    ann_ret = (data["Equity"].iloc[-1] / 100) ** (252 / len(data)) - 1
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() != 0 else 0

    m1.metric("Ann. Return", f"{ann_ret:.2%}")
    m2.metric("Sharpe", f"{sharpe:.2f}")
    m3.metric("Max DD", f"{data['Drawdown'].min():.2%}")
    m4.metric("Hit Ratio", f"{(pd.Series(strat_rets).tail(15) > 0).sum() / 15:.0%}")
    
    st.line_chart(data[["Equity"]])
