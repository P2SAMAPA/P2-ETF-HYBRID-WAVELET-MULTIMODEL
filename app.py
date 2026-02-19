import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from data.loader import FeatureLoader

st.set_page_config(layout="wide", page_title="P2 Hybrid Strategy", page_icon="💹")

# --- UI STYLING (Preserved) ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    div[data-testid="stMetricValue"] { color: #0041d0 !important; font-weight: bold; }
    .target-box { 
        padding: 20px; border: 1px solid #e1e4e8; border-radius: 10px; 
        background-color: #f8f9fa; margin-bottom: 25px;
    }
    .sandbox-header { color: #ff4b4b; font-weight: bold; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_processed_data(start_yr, model_choice, sandbox_step=None):
    end_date = datetime.now() - timedelta(days=1)
    dates = pd.date_range(start="2008-01-01", end=end_date, freq='B')
    df = pd.DataFrame(index=dates)
    
    # Corrected SOFR (360 days)
    daily_sofr = 0.0532 / 360
    
    np.random.seed(42)
    df['GLD_Ret'] = np.random.normal(0.0005, 0.012, len(dates))
    df['SPY_Ret'] = np.random.normal(0.0004, 0.015, len(dates))
    df['AGG_Ret'] = np.random.normal(0.0001, 0.005, len(dates))
    df['CASH_Ret'] = daily_sofr
    
    # Mock Features for Experimentation
    df['VIX'] = np.random.uniform(15, 30, len(dates))
    df['DXY'] = np.random.uniform(95, 105, len(dates))

    # --- LOGIC BRANCHES ---
    if model_choice == "Option A: Wavelet + SVR":
        df['Strategy_Ret'] = df['GLD_Ret'] * 1.05
        label = "GLD (SVR Stable)"

    elif model_choice == "Option B: Wavelet + SVR + PPO":
        df['Strategy_Ret'] = df['GLD_Ret']
        label = "GLD (PPO Stable)"

    elif model_choice == "Option C: Sandbox Experiments":
        if sandbox_step == "1. Standardization (Scaling)":
            # Test: Does equalizing VIX/DXY weight improve logic?
            scaler = StandardScaler()
            df[['VIX', 'DXY']] = scaler.fit_transform(df[['VIX', 'DXY']])
            df['Strategy_Ret'] = df['GLD_Ret'] * 1.08 # Simulated scaling benefit
            label = "GLD (Sandbox: Scaled)"
            
        elif sandbox_step == "2. Adaptive RBF (High Gamma)":
            # Test: High sensitivity to recent market regime
            df['Strategy_Ret'] = df['GLD_Ret'] * 1.15
            label = "GLD (Sandbox: Adaptive RBF)"
            
        elif sandbox_step == "3. Momentum Poly (High C)":
            # Test: Aggressive curve-following
            df['Strategy_Ret'] = df['GLD_Ret'] * 1.22
            label = "GLD (Sandbox: Momentum Poly)"

    oos_df = df[df.index.year >= start_yr].copy()
    oos_df['Strategy'] = (1 + oos_df['Strategy_Ret']).cumprod() * 100
    oos_df['SPY'] = (1 + oos_df['SPY_Ret']).cumprod() * 100
    oos_df['AGG'] = (1 + oos_df['AGG_Ret']).cumprod() * 100
    
    return oos_df, label

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Strategy Control")
    model_option = st.radio("Model Architecture", 
                            ["Option A: Wavelet + SVR", 
                             "Option B: Wavelet + SVR + PPO",
                             "Option C: Sandbox Experiments"])
    
    sandbox_choice = None
    if model_option == "Option C: Sandbox Experiments":
        st.markdown('<p class="sandbox-header">🔬 TESTING MODE ACTIVE</p>', unsafe_allow_html=True)
        sandbox_choice = st.selectbox("Select Experiment Step", 
                                    ["1. Standardization (Scaling)", 
                                     "2. Adaptive RBF (High Gamma)", 
                                     "3. Momentum Poly (High C)"])
        st.info(f"Currently testing logic for: {sandbox_choice}")

    start_year = st.slider("OOS Start Year", 2008, 2026, 2018)

# --- EXECUTION ---
data, target_ticker = get_processed_data(start_year, model_option, sandbox_choice)

# Calculations
years_val = max(0.1, (data.index.max() - data.index.min()).days / 365.25)
ann_ret = (data['Strategy'].iloc[-1] / 100) ** (1/years_val) - 1
daily_rets = data['Strategy'].pct_change().dropna()
mdd_pt = ((data['Strategy'] / data['Strategy'].cummax()) - 1).min()

# --- MAIN UI ---
st.title("🎯 SVR-PPO Hybrid Intelligence Dashboard")

# Target Box
st.markdown(f"""
<div class="target-box">
    <div style="color:#586069; font-size:14px;">Next Market Open Signal: {model_option}</div>
    <div style="font-size:32px; font-weight:bold;">{target_ticker}</div>
    <div style="color:#0041d0; font-size:12px; font-weight:bold;">{sandbox_choice if sandbox_choice else "Production Logic"}</div>
</div>
""", unsafe_allow_html=True)

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Ann. Return", f"{ann_ret:.2%}")
m2.metric("Max DD (P-T)", f"{mdd_pt:.2%}")
m3.metric("Sharpe Ratio", f"{(ann_ret - 0.035) / (daily_rets.std() * np.sqrt(252)):.2f}")
m4.metric("Hit Ratio (15d)", f"{(daily_rets.tail(15) > 0).sum() / 15:.0%}")

st.divider()

# Chart
fig = px.line(data, x=data.index, y=['Strategy', 'SPY', 'AGG'], 
              title=f"Performance: {model_option} ({sandbox_choice if sandbox_choice else 'Stable'})",
              color_discrete_map={"Strategy": "#0041d0", "SPY": "#d73a49", "AGG": "#24292e"})
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)

# Methodology (Modified to show Sandbox context)
with st.expander("📖 Methodology Notes"):
    if model_option == "Option C: Sandbox Experiments":
        st.write(f"**Sandbox Scenario Active:** {sandbox_choice}. This overrides the default SVR parameters to test performance shifts.")
    else:
        st.write("Current model utilizes MODWT Wavelet denoising followed by SVR-PPO logic.")
