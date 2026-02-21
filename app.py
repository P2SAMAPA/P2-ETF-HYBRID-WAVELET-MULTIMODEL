import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data.loader import load_raw_data
from engine import MomentumEngine, A2CEngine, DeepHybridEngine 

st.set_page_config(page_title="P2 ETF WAVELET SVR MULTI MODEL", layout="wide")

@st.cache_data(ttl=3600)
def run_professional_backtest(start_yr, model_choice, t_costs_bps):
    raw_df = load_raw_data()
    assets = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
    t_cost_pct = t_costs_bps / 10_000
    from data.processor import build_feature_matrix

    all_preds = {}
    oos_mask_global = raw_df.index.year >= start_yr
    is_mask_global = raw_df.index.year < start_yr

    for ticker in assets:
        try:
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker)
            oos_m, is_m = idx.year >= start_yr, idx.year < start_yr
            
            if any(opt in model_choice for opt in ["Option I", "Option J", "Option K"]):
                eng = DeepHybridEngine(mode=model_choice)
                lookback = 20
                X_vals = X.values
                oos_indices = np.where(oos_m)[0]
                X_3d = np.array([np.vstack([np.repeat(X_vals[0:1], lookback-len(X_vals[max(0, i-lookback+1):i+1]), axis=0), X_vals[max(0, i-lookback+1):i+1]]) for i in oos_indices])
                X_macro = raw_df[["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]].loc[idx[oos_m]].values if "Option K" in model_choice else None
                preds = eng.predict_series(X_3d, X_macro=X_macro)
            elif "Option C" in model_choice or "Option D: Wavelet-SVR-A2C" in model_choice:
                eng = A2CEngine(); eng.train(X[is_m], y[is_m])
                preds = eng.predict_series(X[oos_m])
            else:
                eng = MomentumEngine(); eng.train(X[is_m], y[is_m])
                preds = eng.predict_series(X[oos_m])
            all_preds[ticker] = pd.Series(preds, index=idx[oos_m])
        except: continue

    df_p = pd.DataFrame(all_preds).dropna()
    oos_idx = raw_df.index[oos_mask_global]
    equity, asset, hwm, timeout = 100.0, "CASH", 100.0, False
    rets, hist, confs = [], [], []

    for d in oos_idx:
        hwm = max(hwm, equity)
        if (equity - hwm)/hwm <= -0.12: timeout = True
        
        dp = df_p.loc[d] if d in df_p.index else None
        raw_sig = dp.idxmax() if (dp is not None and dp.max() > (0.0015 if "Option B" in model_choice else 0)) else "CASH"
        c = 0.5 + (0.18 * ((dp.max()-dp.mean())/dp.std())) if (dp is not None and dp.std()>0) else 0.5
        
        if timeout and c >= 0.75: timeout = False
        final_sig = "CASH" if timeout else raw_sig
        
        if final_sig != asset: equity *= (1 - t_cost_pct); asset = final_sig
        day_r = (raw_df.loc[d, "TBILL_3M"]/100)/252 if asset == "CASH" else raw_df.loc[d, f"{asset}_Ret"]
        equity *= (1 + day_r); rets.append(day_r); hist.append(asset); confs.append(c)

    res = pd.DataFrame({"Strategy_Ret": rets, "Equity": (pd.Series(rets).add(1).cumprod()*100).values}, index=oos_idx)
    res["Drawdown"] = (res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()
    for b in ["SPY", "AGG"]: res[b] = (raw_df.loc[oos_idx, f"{b}_Ret"].add(1).cumprod()*100)
    
    return {"df": res, "audit": pd.DataFrame({"Allocation": hist, "Daily_Return": rets}, index=oos_idx), "target": hist[-1], "confidence": confs[-1], "last_date": oos_idx[-1].strftime('%Y-%m-%d')}

with st.sidebar:
    st.header("Terminal")
    s_yr = st.slider("Year", 2010, 2024, 2015)
    opt = st.radio("Model", ["Option A- Wavelet-SVR", "Option B-Wavelet-SVR-PPO", "Option C: Wavelet-A2C", "Option D: Wavelet-SVR-A2C", "Option D: Wavelet- HMM", "Option E: Wavelet-Bayesian-Regime", "Option G- Wavelet-SVR-HMM", "Option H: Wavelet-SVR-Bayesian", "Option I: Wavelet- CNN-LSTM", "Option J: Wavelet-Attention-CNN-LSTM", "Option K- Wavelet- Parallel-Dual-Stream-CNN-LSTM"])
    costs = st.number_input("Costs", 0, 50, 10)

out = run_professional_backtest(s_yr, opt, costs)
st.markdown(f"### Target: {out['target']} | Confidence: {out['confidence']:.0%}")
st.line_chart(out['df'][['Equity', 'SPY', 'AGG']])
st.subheader("Audit Trail")
st.dataframe(out['audit'].tail(15), use_container_width=True)
