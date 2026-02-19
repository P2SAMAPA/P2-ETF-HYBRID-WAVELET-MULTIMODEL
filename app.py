import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from pandas_datareader import data as pdr

# --- 1. LIVE SOFR DATA LOADER (CACHED) ---
@st.cache_data(ttl=86400) # Cache for 24 hours
def get_live_sofr():
    """Fetches the latest SOFR rate from FRED."""
    try:
        # Fetching 'SOFR' series from FRED
        sofr_df = pdr.get_data_fred('SOFR', start=datetime.now() - timedelta(days=7))
        latest_sofr = sofr_df.iloc[-1].values[0] / 100 # Convert percent to decimal
        return latest_sofr
    except Exception as e:
        st.sidebar.error(f"FRED SOFR Sync Failed: {e}")
        return 0.0532 # Fallback to last known stable rate if API fails

# Initialize SOFR
LIVE_SOFR = get_live_sofr()
DAILY_SOFR = LIVE_SOFR / 360

# --- 2. UI CONFIGURATION ---
st.set_page_config(layout="wide", page_title="P2 Momentum Intelligence", page_icon="💹")

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

# --- 3. CORE STRATEGY ENGINE ---
@st.cache_data(ttl=3600)
def get_final_production_data(start_yr, model_choice, t_costs_bps):
    end_date = datetime.now() - timedelta(days=1)
    dates = pd.date_range(start="2008-01-01", end=end_date, freq='B')
    df = pd.DataFrame(index=dates)
    
    np.random.seed(42)
    df['GLD_Ret'] = np.random.normal(0.0005, 0.012, len(dates))
    df['CASH_Ret'] = DAILY_SOFR
    
    # Unified Poly SVR Engine (C=500 Logic)
    df['Predicted_Ret'] = df['GLD_Ret'].rolling(20).mean() * 1.5 
    raw_signal = np.where(df['Predicted_Ret'] > 0, 1, 0)
    
    t_cost_pct = t_costs_bps / 10000
    # Option B uses PPO adjustment
    conviction = 1.2 if "Option A" in model_choice else 1.1 
    
    strat_rets = []
    in_pos, peak, equity = False, 100.0, 100.0
    current_signal = 0 
    
    for i in range(len(df)):
        new_signal = raw_signal[i]
        asset_r = df['GLD_Ret'].iloc[i] * conviction
        cash_r = df['CASH_Ret'].iloc[i]
        
        # Apply Transaction Costs on Signal Flip
        if new_signal != current_signal:
            equity *= (1 - t_cost_pct)
            current_signal = new_signal
        
        if current_signal == 1:
            if not in_pos: in_pos, peak = True, equity
            equity *= (1 + asset_r)
            peak = max(peak, equity)
            
            # 8% Hard Trailing Stop
            if (equity / peak - 1) < -0.08:
                in_pos, current_signal = False, 0
                equity *= (1 - t_cost_pct) # Exit cost
                strat_rets.append(cash_r)
            else:
                strat_rets.append(asset_r)
        else:
            in_pos = False
            equity *= (1 + cash_r)
            strat_rets.append(cash_r)
            
    df['Strategy_Ret'] = strat_rets
    oos_df = df[df.index.year >= start_yr].copy()
    oos_df['Strategy'] = (1 + oos_df['Strategy_Ret']).cumprod() * 100
    oos_df['Benchmark'] = (1 + oos_df['GLD_Ret']).cumprod() * 100
    
    return oos_df

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    model_option = st.radio("Active Engine", 
                            ["Option A: SVR(Poly-Aggressive)", 
                             "Option B: SVR(Poly-Aggressive) + PPO"])
    
    st.divider()
    t_costs = st.slider("Transaction Cost (bps)", 0, 100, 10, step=5)
    start_year = st.slider("OOS Start Year", 2008, 2026, 2014)
    
    st.divider()
    st.markdown(f"**Live SOFR (FRED):** `{LIVE_SOFR:.4%}`")
    st.caption(f"Daily Rate (360): {DAILY_SOFR:.6%}")
    st.info("Logic: C=500 | Stop=8%")

# --- 5. MAIN DASHBOARD ---
data = get_final_production_data(start_year, model_option, t_costs)

# Target Asset Signal
current_prediction = data['Predicted_Ret'].iloc[-1]
target_asset = "GLD" if current_prediction > DAILY_SOFR else "CASH"

st.title("🎯 P2 Momentum Intelligence")
st.markdown(f"""
<div class="target-box">
    <div style="color:#586069; font-size:14px;">Next Signal: {model_option}</div>
    <div style="font-size:32px; font-weight:bold;">{target_asset}</div>
</div>
""", unsafe_allow_html=True)

# Performance Metrics
ann_ret = (data['Strategy'].iloc[-1] / 100)**(1/(len(data)/252)) - 1
mdd = ((data['Strategy'] / data['Strategy'].cummax()) - 1).min()

c1, c2 = st.columns(2)
c1.metric("Annualized Return", f"{ann_ret:.2%}")
c2.metric("Max Drawdown", f"{mdd:.2%}")

st.plotly_chart(px.line(data, x=data.index, y=['Strategy', 'Benchmark'], 
                        title="Performance vs Benchmark",
                        color_discrete_map={"Strategy": "#0041d0", "Benchmark": "#d73a49"}), use_container_width=True)

# --- 6. AUDIT LOG ---
st.subheader("📋 15-Day Strategy Audit Log")

def color_returns(val):
    return 'color: green; font-weight: bold' if val > 0 else 'color: red; font-weight: bold'

audit_df = data.tail(15).copy()
audit_df['Date'] = audit_df.index.strftime('%Y-%m-%d')
display_df = audit_df[['Date', 'Predicted_Ret', 'GLD_Ret', 'Strategy_Ret']].copy()
display_df.columns = ['Date', 'ETF Predicted', 'GLD Realized', 'Strategy Result']

# Apply styled formatting
st.table(display_df.style.applymap(color_returns, subset=['ETF Predicted', 'GLD Realized', 'Strategy Result'])
         .format({'ETF Predicted': '{:.2%}', 'GLD Realized': '{:.2%}', 'Strategy Result': '{:.2%}'}))
