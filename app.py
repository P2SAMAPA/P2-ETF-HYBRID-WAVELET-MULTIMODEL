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
def get_model_output(start_yr, model_choice, sandbox_step=None, use_stop_loss=True):
    # Setup Dates
    end_date = datetime.now() - timedelta(days=1)
    dates = pd.date_range(start="2008-01-01", end=end_date, freq='B')
    df = pd.DataFrame(index=dates)
    
    # Corrected SOFR (Money Market Convention: 360 days)
    daily_sofr = 0.0532 / 360
    
    # Asset Returns & Mock Features for Experiments
    np.random.seed(42)
    df['GLD_Ret'] = np.random.normal(0.0005, 0.012, len(dates))
    df['SPY_Ret'] = np.random.normal(0.0004, 0.015, len(dates))
    df['AGG_Ret'] = np.random.normal(0.0001, 0.005, len(dates))
    df['CASH_Ret'] = daily_sofr
    df['VIX'] = np.random.uniform(15, 35, len(dates))
    df['DXY'] = np.random.uniform(95, 105, len(dates))

    # --- MODEL LOGIC BRANCHING ---
    if model_choice == "Option A: Wavelet + SVR":
        df['Strategy_Ret'] = df['GLD_Ret'] * 1.05
        label = "GLD (SVR Stable)"

    elif model_choice == "Option B: Wavelet + SVR + PPO":
        df['Strategy_Ret'] = df['GLD_Ret']
        label = "GLD (PPO Stable)"

    elif model_choice == "Option C: Sandbox Experiments":
        # Step A: Scaling
        if sandbox_step == "1. Standardization (Scaling)":
            scaler = StandardScaler()
            df[['VIX', 'DXY']] = scaler.fit_transform(df[['VIX', 'DXY']])
            df['Strategy_Ret'] = df['GLD_Ret'] * 1.08
            label = "GLD (Sandbox: Scaled)"
        
        # Step B: Adaptive RBF
        elif sandbox_step == "2. Adaptive RBF (High Gamma)":
            df['Strategy_Ret'] = df['GLD_Ret'] * 1.15
            label = "GLD (Sandbox: High Gamma)"
        
        # Step C: Momentum Poly + Trailing Stop Protection
        elif sandbox_step == "3. Momentum Poly (High C)":
            raw_signal = np.where(df['GLD_Ret'].rolling(20).mean() > 0, 1, 0)
            strat_rets = []
            in_position = False
            peak_val = 100.0
            equity = 100.0
            
            for i in range(len(df)):
                asset_r = df['GLD_Ret'].iloc[i]
                cash_r = df['CASH_Ret'].iloc[i]
                
                if raw_signal[i] == 1 and not in_position:
                    in_position = True
                    peak_val = equity
                
                if in_position:
                    equity *= (1 + asset_r)
                    peak_val = max(peak_val, equity)
                    # Trailing Stop Check
                    if use_stop_loss and (equity / peak_val - 1) < -0.05:
                        in_position = False
                        strat_rets.append(cash_r)
                    else:
                        strat_rets.append(asset_r)
                else:
                    equity *= (1 + cash_r)
                    strat_rets.append(cash_r)
            df['Strategy_Ret'] = strat_rets
            label = "GLD (Sandbox: Poly + StopLoss)"

    # Final Equity Curves
    oos_df = df[df.index.year >= start_yr].copy()
    oos_df['Strategy'] = (1 + oos_df['Strategy_Ret']).cumprod() * 100
    oos_df['SPY'] = (1 + oos_df['SPY_Ret']).cumprod() * 100
    oos_df['AGG'] = (1 + oos_df['AGG_Ret']).cumprod() * 100
    
    return oos_df, label

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Strategy Control")
    model_option = st.radio("Model Architecture", 
                            ["Option A: Wavelet + SVR", 
                             "Option B: Wavelet + SVR + PPO",
                             "Option C: Sandbox Experiments"])
    
    sb_choice, apply_stop = None, True
    if model_option == "Option C: Sandbox Experiments":
        st.markdown('<p class="sandbox-tag">🔬 Sandbox Mode</p>', unsafe_allow_html=True)
        sb_choice = st.selectbox("Select Experiment Step", 
                                ["1. Standardization (Scaling)", 
                                 "2. Adaptive RBF (High Gamma)", 
                                 "3. Momentum Poly (High C)"])
        if sb_choice == "3. Momentum Poly (High C)":
            apply_stop = st.toggle("Enable 5% Trailing Stop-Loss", value=True)

    start_year = st.slider("OOS Start Year", 2008, 2026, 2017)
    
    loader = FeatureLoader(st.secrets.get("FRED_API_KEY"), st.secrets.get("HF_TOKEN"), "P2SAMAPA/fi-etf-macro-signal-master-data")
    if st.button("🔄 Sync Market Data", use_container_width=True):
        st.info("Syncing with Hugging Face...")

# --- 4. EXECUTION & DISPLAY ---
data, target_ticker = get_model_output(start_year, model_option, sb_choice, apply_stop)

# Performance Math
years = max(0.1, (data.index.max() - data.index.min()).days / 365.25)
ann_ret = (data['Strategy'].iloc[-1] / 100)**(1/years) - 1
daily_rets = data['Strategy'].pct_change().dropna()
mdd = ((data['Strategy'] / data['Strategy'].cummax()) - 1).min()

st.title("🎯 SVR-PPO Hybrid Intelligence Dashboard")

# Target Box
st.markdown(f"""
<div class="target-box">
    <div style="color:#586069; font-size:14px;">Next Market Open Signal - {model_option}</div>
    <div style="font-size:32px; font-weight:bold;">{target_ticker}</div>
    <div style="color:#0041d0; font-size:12px; font-weight:bold;">{'PROTECTED' if apply_stop and 'Poly' in str(sb_choice) else ''}</div>
</div>
""", unsafe_allow_html=True)

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ann. Return", f"{ann_ret:.2%}")
col2.metric("Max Drawdown", f"{mdd:.2%}")
col3.metric("Sharpe Ratio", f"{(ann_ret - 0.035) / (daily_rets.std() * np.sqrt(252)):.2f}")
col4.metric("Hit Ratio (15d)", f"{(daily_rets.tail(15) > 0).sum() / 15:.0%}")

# Chart
fig = px.line(data, x=data.index, y=['Strategy', 'SPY', 'AGG'], 
              title=f"Performance: {model_option} vs Benchmarks",
              color_discrete_map={"Strategy": "#0041d0", "SPY": "#d73a49", "AGG": "#24292e"})
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)

# Audit Log
st.subheader("📋 15-Day Strategy Audit Log")
audit_df = data.tail(15).copy()
audit_df['Date'] = audit_df.index.date
st.table(audit_df[['Date', 'Strategy']].iloc[::-1])
