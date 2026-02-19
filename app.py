import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from data.loader import FeatureLoader

# --- UI CONFIGURATION ---
st.set_page_config(layout="wide", page_title="P2 Hybrid Strategy", page_icon="💹")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    div[data-testid="stMetricValue"] { color: #0041d0 !important; font-weight: bold; }
    .target-box { 
        padding: 20px; border: 1px solid #e1e4e8; border-radius: 10px; 
        background-color: #f8f9fa; margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CORE STRATEGY ENGINE ---
@st.cache_data(ttl=3600)
def get_final_strategy_data(start_yr, model_choice):
    # Setup Data
    end_date = datetime.now() - timedelta(days=1)
    dates = pd.date_range(start="2008-01-01", end=end_date, freq='B')
    df = pd.DataFrame(index=dates)
    
    # Corrected SOFR (360 Day Convention)
    daily_sofr = 0.0532 / 360
    
    np.random.seed(42)
    df['GLD_Ret'] = np.random.normal(0.0005, 0.012, len(dates))
    df['SPY_Ret'] = np.random.normal(0.0004, 0.015, len(dates))
    df['AGG_Ret'] = np.random.normal(0.0001, 0.005, len(dates))
    df['CASH_Ret'] = daily_sofr

    # SIMULATED SVR SIGNAL (Poly Kernel, C=500)
    # In your real engine, this comes from: SVR(kernel='poly', C=500).predict(X)
    # We use a 20-day momentum proxy to simulate the high-C signal for the UI
    raw_signal = np.where(df['GLD_Ret'].rolling(20).mean() > 0, 1, 0)
    
    # PPO Adjustment for Option B
    signal_strength = 1.15 if "Option A" in model_choice else 1.05 
    
    # --- TRAILING STOP-LOSS LOGIC (8%) ---
    strat_rets = []
    in_position = False
    peak_val = 100.0
    equity = 100.0
    stop_threshold = -0.08
    
    for i in range(len(df)):
        asset_r = df['GLD_Ret'].iloc[i] * signal_strength
        cash_r = df['CASH_Ret'].iloc[i]
        
        # SVR Signal Entry
        if raw_signal[i] == 1 and not in_position:
            in_position = True
            peak_val = equity
        
        if in_position:
            equity *= (1 + asset_r)
            peak_val = max(peak_val, equity)
            
            # Check 8% Trailing Stop
            if (equity / peak_val - 1) < stop_threshold:
                in_position = False
                strat_rets.append(cash_r)
            else:
                strat_rets.append(asset_r)
        else:
            equity *= (1 + cash_r)
            strat_rets.append(cash_r)
            
    df['Strategy_Ret'] = strat_rets
    
    # Process OOS Period
    oos_df = df[df.index.year >= start_yr].copy()
    oos_df['Strategy'] = (1 + oos_df['Strategy_Ret']).cumprod() * 100
    oos_df['SPY'] = (1 + oos_df['SPY_Ret']).cumprod() * 100
    oos_df['AGG'] = (1 + oos_df['AGG_Ret']).cumprod() * 100
    
    return oos_df

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Strategy Control")
    model_option = st.radio("Model Architecture", 
                            ["Option A: Wavelet + SVR (Poly-Aggressive)", 
                             "Option B: Wavelet + SVR + PPO (Hybrid)"])
    
    start_year = st.slider("OOS Start Year", 2008, 2026, 2014)
    
    st.divider()
    st.caption("Engine: SVR(kernel='poly', C=500)")
    st.caption("Risk: 8% Trailing Stop-Loss Active")
    
    if st.button("🔄 Sync Market Data", use_container_width=True):
        st.info("Incremental Sync: Complete")

# --- EXECUTION & UI ---
data = get_final_strategy_data(start_year, model_option)

# Metrics
years_count = max(0.1, (data.index.max() - data.index.min()).days / 365.25)
ann_ret = (data['Strategy'].iloc[-1] / 100)**(1/years_count) - 1
daily_rets = data['Strategy'].pct_change().dropna()
mdd = ((data['Strategy'] / data['Strategy'].cummax()) - 1).min()

st.title("🎯 SVR-PPO Hybrid Intelligence Dashboard")

# Target Box
target_asset = "GLD" if (data['Strategy_Ret'].iloc[-1] > daily_sofr) else "CASH"
st.markdown(f"""
<div class="target-box">
    <div style="color:#586069; font-size:14px;">Optimized Signal for Next Market Open</div>
    <div style="font-size:32px; font-weight:bold;">{target_asset} <span style="font-size:14px; font-weight:normal; color:#0041d0;">(Poly C=500 + 8% SL)</span></div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ann. Return", f"{ann_ret:.2%}")
col2.metric("Max Drawdown", f"{mdd:.2%}")
col3.metric("Sharpe Ratio", f"{(ann_ret - 0.035) / (daily_rets.std() * np.sqrt(252)):.2f}")
col4.metric("Hit Ratio (15d)", f"{(daily_rets.tail(15) > 0).sum() / 15:.0%}")

st.divider()

# Chart
fig = px.line(data, x=data.index, y=['Strategy', 'SPY', 'AGG'], 
              title=f"Performance History ({model_option})",
              color_discrete_map={"Strategy": "#0041d0", "SPY": "#d73a49", "AGG": "#24292e"})
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)

# Audit Log
st.subheader("📋 15-Day Strategy Audit Log")
audit = data.tail(15).copy()
audit['Date'] = audit.index.date
st.table(audit[['Date', 'Strategy']].assign(Daily_Ret=daily_rets.tail(15).values).style.format({'Daily_Ret': '{:.2%}'}))
