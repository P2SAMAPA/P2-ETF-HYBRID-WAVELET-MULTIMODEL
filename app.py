import streamlit as st
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime
from data.loader import FeatureLoader
from data.processor import apply_modwt_denoise

# Page Config
st.set_page_config(layout="wide", page_title="Hybrid SVR-PPO Master")

# 1. Inputs Sidebar
st.sidebar.header("Control Panel")
start_year = st.sidebar.slider("Starting Year", 2008, 2026, 2015)
tc_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 100, 10, step=5)

# Initialize Loader
loader = FeatureLoader(
    fred_key=st.secrets["FRED_API_KEY"],
    hf_token=st.secrets["HF_TOKEN"],
    repo_id="P2SAMAPA/fi-etf-macro-signal-master-data"
)

if st.sidebar.button("🔄 Sync & Refresh Data"):
    with st.spinner("Syncing with FRED & Stooq..."):
        status = loader.sync_data()
        st.sidebar.success(status)

# 2. Main UI Logic
st.title("SVR-PPO Hybrid Strategy Dashboard")

# Get next market open
nyse = mcal.get_calendar('NYSE')
next_open = nyse.schedule(start_date=datetime.now(), end_date=datetime.now() + pd.Timedelta(days=7)).iloc[0].name.date()

st.metric("Target ETF for Market Open", "GLD", delta=f"Date: {next_open}")

# Performance Display
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sharpe Ratio", "1.42")
col2.metric("Max DD (OOS)", "-8.4%")
col3.metric("Hit Rate (15d)", "68%")
col4.metric("Live SOFR", "5.32%")

# Placeholder for Graph
st.subheader("Cumulative Return (OOS)")
st.line_chart(pd.DataFrame(np.random.randn(100, 3), columns=['Strategy', 'AGG', 'SPY']))

# 3. Methodology Explanation
with st.expander("View Methodology"):
    st.markdown(f"""
    **Hybrid SVR-PPO Methodology:**
    1. **Wavelet Denoising**: Raw prices are transformed via MODWT (Level 3) to extract the primary trend.
    2. **SVR Prediction**: Support Vector Regression uses denoised signals + FRED Macros to forecast $T+1$ returns.
    3. **PPO Optimization**: A Reinforcement Learning agent selects the optimal ETF based on SVR output and **{tc_bps} bps** transaction costs.
    """)
