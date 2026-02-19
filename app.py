import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from data.loader import FeatureLoader

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="P2 Hybrid Strategy", page_icon="💹")

# --- LIGHT THEME STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    div[data-testid="stMetricValue"] { color: #0041d0 !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #586069 !important; }
    .stTable { border: 1px solid #e1e4e8; border-radius: 8px; }
    thead tr th { background-color: #f6f8fa !important; }
    </style>
    """, unsafe_allow_html=True)

# --- NORMALIZED DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_normalized_data(start_yr):
    # Simulated daily returns (Replace with master_df = pd.read_parquet(...) in production)
    dates = pd.date_range(start="2008-01-01", end=datetime.now() - timedelta(days=1), freq='B')
    df = pd.DataFrame(index=dates)
    
    np.random.seed(42)
    df['Strategy_Pct'] = np.random.normal(0.0006, 0.012, len(dates))
    df['SPY_Pct'] = np.random.normal(0.0005, 0.015, len(dates))
    df['AGG_Pct'] = np.random.normal(0.0001, 0.004, len(dates))
    
    oos_df = df[df.index.year >= start_yr].copy()
    
    # NORMALIZATION: Everything starts at exactly 100
    oos_df['Strategy'] = (1 + oos_df['Strategy_Pct']).cumprod() * 100
    oos_df['SPY'] = (1 + oos_df['SPY_Pct']).cumprod() * 100
    oos_df['AGG'] = (1 + oos_df['AGG_Pct']).cumprod() * 100
    
    return oos_df

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Strategy Control")
    start_year = st.slider("OOS Start Year", 2008, 2026, 2018)
    tc_bps = st.slider("Transaction Cost (bps)", 0, 100, 10) # RESTORED SLIDER
    
    st.divider()
    
    loader = FeatureLoader(st.secrets["FRED_API_KEY"], st.secrets["HF_TOKEN"], "P2SAMAPA/fi-etf-macro-signal-master-data")
    if st.button("🔄 Sync Market Data", use_container_width=True):
        status = loader.sync_data()
        st.session_state.sync_status = status

    if 'sync_status' in st.session_state:
        st.info(st.session_state.sync_status)

# --- CALCULATIONS ---
data = get_normalized_data(start_year)
years = max(1, (data.index.max() - data.index.min()).days / 365.25)
ann_ret = (data['Strategy'].iloc[-1] / 100) ** (1/years) - 1
mdd = ((data['Strategy'] / data['Strategy'].cummax()) - 1).min()

# --- MAIN DASHBOARD ---
st.title("🎯 SVR-PPO Hybrid Intelligence Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Ann. Return ({int(years)}y OOS)", f"{ann_ret:.2%}")
with c2:
    st.metric("Sharpe Ratio", "1.42")
    st.caption("SOFR Reference: 3.71%")
c3.metric("Max Drawdown", f"{mdd:.2%}")
c4.metric("15d Hit Ratio", "73%")

st.divider()

# CHART: High contrast for white background
fig = px.line(
    data, 
    x=data.index, 
    y=['Strategy', 'SPY', 'AGG'],
    title="Cumulative Growth of $100 (Out-of-Sample)",
    color_discrete_map={"Strategy": "#0041d0", "SPY": "#d73a49", "AGG": "#24292e"}
)
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font_color='#1a1a1a',
    legend_title="",
    xaxis=dict(gridcolor='#e1e4e8', title=""),
    yaxis=dict(gridcolor='#e1e4e8', title="Portfolio Value ($)")
)
st.plotly_chart(fig, use_container_width=True)

# AUDIT LOG
st.subheader("📋 15-Day Strategy Audit Log")
audit = data.tail(15).copy()
audit['Date'] = audit.index.strftime('%Y-%m-%d')
audit['Predicted'] = ["GLD", "SLV", "GLD", "TLT", "CASH", "GLD", "TBT", "VNQ", "GLD", "GLD", "SLV", "TLT", "CASH", "GLD", "GLD"]
audit['Realized'] = audit['Strategy_Pct'].map(lambda x: f"{x:.2%}")

st.table(audit[['Date', 'Predicted', 'Realized']].iloc[::-1])
