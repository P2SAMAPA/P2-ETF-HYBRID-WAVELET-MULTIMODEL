import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data.loader import load_raw_data
from engine import MomentumEngine, A2CEngine, DeepHybridEngine 

# --- CONFIG & THEME ---
st.set_page_config(page_title="P2 ETF WAVELET SVR MULTI MODEL", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; }
    [data-testid="stMetricValue"] { color: #1a73e8 !important; font-size: 22px !important; }
    [data-testid="stMetricLabel"] { font-weight: 700; text-transform: uppercase; font-size: 10px; color: #5f6368 !important; }
    /* Styling for the new Max DD Daily box */
    .dd-box { background-color: #fff5f5; padding: 12px; border-radius: 8px; border: 1px solid #feb2b2; text-align: center; }
    </style>
""", unsafe_allow_html=True)

def get_next_trading_day_simple():
    now = datetime.now()
    # Move to next weekday logic for NYSE
    if now.weekday() >= 4: 
        days_ahead = (7 - now.weekday()) if now.weekday() > 4 else 3
        next_day = now + timedelta(days=days_ahead)
    else:
        next_day = now + timedelta(days=1)
    return next_day.strftime('%d %B %Y')

@st.cache_data(ttl=3600)
def run_professional_backtest(start_yr, model_choice, t_costs_bps, _force_refresh=None):
    # Pass force_sync into your loader logic
    raw_df = load_raw_data(force_sync=(_force_refresh is not None))
    assets = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
    t_cost_pct = t_costs_bps / 10_000
    from data.processor import build_feature_matrix

    all_preds = {}
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
    
    # EXECUTION LOOP WITH 18% HWM RISK GATE
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

    res = pd.DataFrame({"Equity": (pd.Series(rets).add(1).cumprod()*100).values}, index=common_idx)
    res["Strategy_Ret"] = rets
    res["Drawdown"] = (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    res["RF"] = (raw_df.loc[common_idx, "TBILL_3M"] / 100) / 252
    
    for b in ["SPY", "AGG"]:
        res[b] = (raw_df.loc[common_idx, f"{b}_Ret"] + 1).cumprod() * 100
    
    return {"df": res, "audit": pd.DataFrame({"Allocation": hist, "Return": rets}, index=common_idx), 
            "target": hist[-1], "conf": confs[-1], "date": common_idx[-1].strftime('%Y-%m-%d')}

# --- SIDEBAR & INCREMENTAL TRIGGER
