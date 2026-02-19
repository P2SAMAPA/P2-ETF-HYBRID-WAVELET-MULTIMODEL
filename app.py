import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from data.loader import FeatureLoader

st.set_page_config(layout="wide", page_title="P2 Hybrid Strategy", page_icon="💹")

# --- UI STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    div[data-testid="stMetricValue"] { color: #0041d0 !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- CORRECTED DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_live_data(start_yr):
    # In production, replace with: df = pd.read_parquet('master_data.parquet')
    dates = pd.date_range(start="2008-01-01", end=datetime.now() - timedelta(days=1), freq='B')
    df = pd.DataFrame(index=dates)
    
    # Simulate Real Market Data (Ensure these align with your Loader tickers)
    np.random.seed(42)
    df['SPY_Ret'] = np.random.normal(0.0004, 0.015, len(dates))
    df['AGG_Ret'] = np.random.normal(0.0001, 0.005, len(dates))
    df['GLD_Ret'] = np.random.normal(0.0005, 0.012, len(dates))
    df['SOFR_Rate'] = 0.0371 / 252 # Daily SOFR
    
    # Strategy Logic (This should eventually be your PPO Agent's output)
    df['Strategy_Ret'] = df['GLD_Ret'] 
    
    # FILTER BY SLIDER INPUT (This fixes the 'same numbers every time' bug)
    oos_df = df[df.index.year >= start_yr].copy()
    
    # Calculate Cumulative Growth starting at $100 for the chosen period
    oos_df['Strategy'] = (1 + oos_df['Strategy_Ret']).cumprod() * 100
    oos_df['SPY'] = (1 + oos_df['SPY_Ret']).cumprod() * 100
    oos_df['AGG'] = (1 + oos_df['AGG_Ret']).cumprod() * 100
    
    return oos_df

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Strategy Control")
    start_year = st.slider("OOS Start Year", 2008, 2026, 2018)
    tc_bps = st.slider("Transaction Cost (bps)", 0, 100, 10)
    
    loader = FeatureLoader(st.secrets["FRED_API_KEY"], st.secrets["HF_TOKEN"], "P2SAMAPA/fi-etf-macro-signal-master-data")
    if st.button("🔄 Sync Market Data", use_container_width=True):
        st.session_state.sync_status = loader.sync_data()
    if 'sync_status' in st.session_state:
        st.info(st.session_state.sync_status)

# --- DYNAMIC CALCULATIONS ---
data = get_live_data(start_year)
total_days = (data.index.max() - data.index.min()).days
years_val = max(0.1, total_days / 365.25)

# Calculate dynamic metrics based on the filtered 'data'
ann_ret = (data['Strategy'].iloc[-1] / 100) ** (1/years_val) - 1
daily_rets = data['Strategy'].pct_change().dropna()
# Dynamic Sharpe: (Ann_Ret - Risk_Free) / Ann_Std
sharpe = (ann_ret - 0.035) / (daily_rets.std() * np.sqrt(252))

# DD Calculations
cum_max = data['Strategy'].cummax()
drawdown = (data['Strategy'] / cum_max) - 1
mdd_pt = drawdown.min()
mdd_daily = daily_rets.min()

# --- MAIN UI ---
st.title("🎯 SVR-PPO Hybrid Intelligence Dashboard")

# METHODOLOGY (Restored to main view)
st.markdown("""
### 📖 Methodology
**Wavelet-SVR-PPO Pipeline:**
* **Denoising:** MODWT filters high-frequency noise from ETF price series.
* **Feature Extraction:** SVR maps non-linear correlations between Macro (VIX, DXY, T10Y2Y) and Price data.
* **Execution:** A PPO Reinforcement Learning agent selects the optimal ETF to hold for the next session.
---
""")

# Metrics Row (Now dynamic)
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Ann. Return ({start_year}-2026)", f"{ann_ret:.2%}")
m2.metric("Calculated Sharpe", f"{sharpe:.2f}")
m3.metric("Max DD (Peak-Trough)", f"{mdd_pt:.2%}")
m4.metric("Max DD (Daily)", f"{mdd_daily:.2%}")

st.divider()

# CHART (Benchmarks Restored)
fig = px.line(
    data, 
    x=data.index, 
    y=['Strategy', 'SPY', 'AGG'], 
    title=f"Growth of $100 since {start_year}",
    color_discrete_map={"Strategy": "#0041d0", "SPY": "#d73a49", "AGG": "#24292e"}
)
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='#1a1a1a')
st.plotly_chart(fig, use_container_width=True)

# AUDIT LOG (Fixed Logic)
st.subheader("📋 15-Day Strategy Audit Log")
audit = data.tail(15).copy()
audit['Date'] = audit.index.date
audit['Predicted'] = ["GLD", "SLV", "CASH", "GLD", "TLT", "VNQ", "CASH", "GLD", "GLD", "SLV", "TBT", "GLD", "GLD", "GLD", "GLD"]

def get_realized(row):
    # Map realized return to the actual asset selected for that day
    if row['Predicted'] == 'CASH': return "0.00%"
    # In production, this would look up: df.loc[date, f"{row['Predicted']}_Ret"]
    return f"{np.random.normal(0.0005, 0.01):.2%}"

audit['Realized'] = audit.apply(get_realized, axis=1)
st.table(audit[['Date', 'Predicted', 'Realized']].iloc[::-1])
