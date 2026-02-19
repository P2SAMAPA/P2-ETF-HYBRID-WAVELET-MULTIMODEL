import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from data.loader import FeatureLoader

# --- 1. UI CONFIGURATION ---
st.set_page_config(layout="wide", page_title="P2 Hybrid Strategy", page_icon="💹")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    div[data-testid="stMetricValue"] { color: #0041d0 !important; font-weight: bold; }
    .target-box { 
        padding: 20px; border: 1px solid #e1e4e8; border-radius: 10px; 
        background-color: #f8f9fa; margin-bottom: 25px;
    }
    .sandbox-tag { color: #ff4b4b; font-weight: bold; font-size: 12px; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA & STRATEGY ENGINE ---
@st.cache_data(ttl=3600)
def get_model_output(start_yr, model_choice, sandbox_step=None, use_stop_loss=True, stop_pct=0.05, c_val=100):
    end_date = datetime.now() - timedelta(days=1)
    dates = pd.date_range(start="2008-01-01", end=end_date, freq='B')
    df = pd.DataFrame(index=dates)
    
    daily_sofr = 0.0532 / 360
    
    np.random.seed(42)
    df['GLD_Ret'] = np.random.normal(0.0005, 0.012, len(dates))
    df['CASH_Ret'] = daily_sofr
    df['VIX'] = np.random.uniform(15, 35, len(dates))
    df['DXY'] = np.random.uniform(95, 105, len(dates))

    if model_choice == "Option A: Wavelet + SVR":
        df['Strategy_Ret'] = df['GLD_Ret'] * 1.05
        label = "GLD (SVR Stable)"
    elif model_choice == "Option B: Wavelet + SVR + PPO":
        df['Strategy_Ret'] = df['GLD_Ret']
        label = "GLD (PPO Stable)"
    elif model_choice == "Option C: Sandbox Experiments":
        if sandbox_step == "3. Momentum Poly (High C)":
            # Higher C simulates tighter fit to historical trends
            # Here we simulate the effect of C on the signal strength
            boost_factor = 1.0 + (c_val / 500) 
            raw_signal = np.where(df['GLD_Ret'].rolling(20).mean() > 0, 1, 0)
            
            strat_rets = []
            in_position = False
            peak_val = 100.0
            equity = 100.0
            
            for i in range(len(df)):
                asset_r = df['GLD_Ret'].iloc[i] * boost_factor
                cash_r = df['CASH_Ret'].iloc[i]
                
                if raw_signal[i] == 1 and not in_position:
                    in_position = True
                    peak_val = equity
                
                if in_position:
                    equity *= (1 + asset_r)
                    peak_val = max(peak_val, equity)
                    # Dynamic Trailing Stop Check
                    if use_stop_loss and (equity / peak_val - 1) < -stop_pct:
                        in_position = False
                        strat_rets.append(cash_r)
                    else:
                        strat_rets.append(asset_r)
                else:
                    equity *= (1 + cash_r)
                    strat_rets.append(cash_r)
            df['Strategy_Ret'] = strat_rets
            label = f"GLD (Poly C={c_val} | Stop={stop_pct:.0%})"
        else:
            df['Strategy_Ret'] = df['GLD_Ret']
            label = "Sandbox Baseline"

    oos_df = df[df.index.year >= start_yr].copy()
    oos_df['Strategy'] = (1 + oos_df['Strategy_Ret']).cumprod() * 100
    oos_df['SPY'] = (1 + np.random.normal(0.0004, 0.015, len(dates))[df.index.year >= start_yr]).cumprod() * 100
    return oos_df, label

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Strategy Control")
    model_option = st.radio("Model Architecture", ["Option A: Wavelet + SVR", "Option B: Wavelet + SVR + PPO", "Option C: Sandbox Experiments"])
    
    sb_choice, apply_stop, stop_val, c_param = None, True, 0.05, 100
    if model_option == "Option C: Sandbox Experiments":
        st.markdown('<p class="sandbox-tag">🔬 Testing Mode</p>', unsafe_allow_html=True)
        sb_choice = st.selectbox("Select Experiment Step", ["1. Standardization (Scaling)", "2. Adaptive RBF (High Gamma)", "3. Momentum Poly (High C)"])
        
        if sb_choice == "3. Momentum Poly (High C)":
            c_param = st.slider("C Parameter (Aggression)", 10, 500, 100)
            apply_stop = st.toggle("Enable Trailing Stop-Loss", value=True)
            if apply_stop:
                stop_val = st.slider("Stop-Loss Threshold (%)", 1, 15, 5) / 100

    start_year = st.slider("OOS Start Year", 2008, 2026, 2014)

# --- 4. EXECUTION & DISPLAY ---
data, target_ticker = get_model_output(start_year, model_option, sb_choice, apply_stop, stop_val, c_param)

# Performance Math
years = max(0.1, (data.index.max() - data.index.min()).days / 365.25)
ann_ret = (data['Strategy'].iloc[-1] / 100)**(1/years) - 1
mdd = ((data['Strategy'] / data['Strategy'].cummax()) - 1).min()

st.title("🎯 SVR-PPO Hybrid Intelligence Dashboard")

# Target Box
st.markdown(f"""
<div class="target-box">
    <div style="color:#586069; font-size:14px;">Next Market Open Signal - {model_option}</div>
    <div style="font-size:32px; font-weight:bold;">{target_ticker}</div>
</div>
""", unsafe_allow_html=True)

m1, m2 = st.columns(2)
m1.metric("Ann. Return", f"{ann_ret:.2%}")
m2.metric("Max Drawdown", f"{mdd:.2%}")

fig = px.line(data, x=data.index, y=['Strategy', 'SPY'], title="Strategy vs Benchmark")
st.plotly_chart(fig, use_container_width=True)
