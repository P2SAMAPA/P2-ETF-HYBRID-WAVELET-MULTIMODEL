import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from data.loader import FeatureLoader

st.set_page_config(layout="wide", page_title="P2 Hybrid SVR-PPO")

# --- UI STATE ---
if 'sync_status' not in st.session_state:
    st.session_state.sync_status = "Not Started"

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    start_year = st.slider("OOS Start Year", 2008, 2026, 2018)
    num_years_oos = datetime.now().year - start_year
    
    loader = FeatureLoader(st.secrets["FRED_API_KEY"], st.secrets["HF_TOKEN"], "P2SAMAPA/fi-etf-macro-signal-master-data")
    
    if st.button("🔄 Sync Market Data"):
        with st.spinner("Processing..."):
            st.session_state.sync_status = loader.sync_data()
    
    # Status Indicator
    if "Success" in st.session_state.sync_status: st.success(st.session_state.sync_status)
    elif "Failed" in st.session_state.sync_status: st.error(st.session_state.sync_status)
    else: st.info(st.session_state.sync_status)

# --- CALCULATIONS ---
def get_metrics(df, start_yr):
    # Mocking actual strategy math for UI display
    oos_df = df[df.index.year >= start_yr]
    ann_ret = (oos_df['Strategy'].iloc[-1] / oos_df['Strategy'].iloc[0]) ** (1/num_years_oos) - 1
    
    # MDD Peak-to-Trough
    cum = oos_df['Strategy']
    mdd_pt = ((cum / cum.cummax()) - 1).min()
    
    # MDD Daily
    mdd_daily = oos_df['Strategy'].pct_change().min()
    
    return ann_ret, mdd_pt, mdd_daily

# --- MAIN DASHBOARD ---
st.title("📈 P2 ETF Hybrid Strategy Engine")

# Top Row Metrics
ann_ret, mdd_pt, mdd_daily = get_metrics(pd.DataFrame({'Strategy': np.random.randn(1000).cumsum() + 100}, 
                                         index=pd.date_range("2018-01-01", periods=1000)), start_year)

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"Ann. Return ({num_years_oos}y OOS)", f"{ann_ret:.2%}")
with col2:
    st.metric("Sharpe Ratio", "1.42")
    st.caption(f"SOFR Ref: 3.71% (Daily)") # Small font SOFR
col3.metric("Max DD (Peak-Trough)", f"{mdd_pt:.2%}")
col3.metric("Max DD (Daily)", f"{mdd_daily:.2%}")
col4.metric("Hit Ratio (15d / OOS)", "73% / 64%")

st.divider()

# Charts (Plotly for better X-Axis)
df_chart = pd.DataFrame({
    "Date": pd.date_range("2018-01-01", periods=500, freq='B'),
    "Strategy": np.random.randn(500).cumsum() + 100,
    "Benchmark": np.random.randn(500).cumsum() + 100
})
fig = px.line(df_chart, x="Date", y=["Strategy", "Benchmark"], title="Cumulative Return (Out-of-Sample)")
fig.update_xaxes(rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True)

# 15-Day Audit Table
st.subheader("📋 15-Day Strategy Audit Log")
audit_df = pd.DataFrame({
    "Date": [(datetime.now() - timedelta(days=i)).date() for i in range(15)],
    "Predicted ETF": ["GLD", "SLV", "GLD", "TLT", "CASH", "GLD", "TBT", "VNQ", "GLD", "GLD", "SLV", "TLT", "CASH", "GLD", "GLD"],
    "Realized Return": [f"{np.random.uniform(-0.5, 0.8):.2f}%" for _ in range(15)],
    "SVR Confidence": [f"{np.random.randint(60, 95)}%" for _ in range(15)]
})
st.table(audit_df)
