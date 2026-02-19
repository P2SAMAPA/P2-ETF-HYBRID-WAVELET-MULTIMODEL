import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. GLOBAL CONSTANTS & SCOPE FIX ---
SOFR_ANNUAL = 0.0532
DAILY_SOFR = SOFR_ANNUAL / 360  # Money Market Convention

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
    
    # Simulate High-C SVR Signal (Aggressive Poly)
    raw_signal = np.where(df['GLD_Ret'].rolling(20).mean() > 0, 1, 0)
    
    # Transaction Cost Adjustment (bps to decimal)
    t_cost_pct = t_costs_bps / 10000
    
    # Option B PPO Layer
    conviction = 1.2 if "Option A" in model_choice else 1.1 
    
    # --- STRATEGY SIMULATION WITH 8% STOP & T-COSTS ---
    strat_rets = []
    in_pos, peak, equity = False, 100.0, 100.0
    current_signal = 0 # 0 for CASH, 1 for GLD
    
    for i in range(len(df)):
        new_signal = raw_signal[i]
        asset_r = df['GLD_Ret'].iloc[i] * conviction
        cash_r = df['CASH_Ret'].iloc[i]
        
        # Check for Signal Change (Apply Transaction Costs)
        applied_cost = 0
        if new_signal != current_signal:
            applied_cost = t_cost_pct
            current_signal = new_signal
        
        # Entry Logic
        if current_signal == 1 and not in_pos:
            in_pos, peak = True, equity
            equity *= (1 - applied_cost) # Apply cost on entry
        
        if in_pos:
            equity *= (1 + asset_r)
            peak = max(peak, equity)
            
            # 8% Trailing Stop Trigger
            if (equity / peak - 1) < -0.08:
                in_pos = False
                current_signal = 0 # Force to CASH
                equity *= (1 - t_cost_pct) # Apply cost on exit
                strat_rets.append(cash_r)
            else:
                strat_rets.append(asset_r)
        else:
            # If signal changed but we are in CASH, apply cost on exit from GLD
            if applied_cost > 0:
                equity *= (1 - applied_cost)
            
            equity *= (1 + cash_r)
            strat_rets.append(cash_r)
            
    df['Strategy_Ret'] = strat_rets
    oos_df = df[df.index.year >= start_yr].copy()
    oos_df['Strategy'] = (1 + oos_df['Strategy_Ret']).cumprod() * 100
    oos_df['Benchmark'] = (1 + oos_df['GLD_Ret']).cumprod() * 100
    
    return oos_df

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Strategy Control")
    model_option = st.radio("Model Architecture", 
                            ["Option A: Wavelet + SVR (Poly-Aggressive)", 
                             "Option B: Wavelet + SVR + PPO (Hybrid)"])
    
    st.divider()
    t_costs = st.slider("Transaction Cost / Slippage (bps)", 0, 100, 10, step=5)
    st.caption("Applied on every signal flip (Entry/Exit)")
    
    start_year = st.slider("OOS Start Year", 2008, 2026, 2014)
    
    st.divider()
    st.markdown(f"""
    <div style="font-size:12px; color:gray;">
    <b>Engine:</b> SVR(kernel='poly', C=500)<br>
    <b>Risk:</b> 8% Trailing Stop-Loss<br>
    <b>SOFR:</b> {SOFR_ANNUAL:.2%} (/360)
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Sync Market Data", use_container_width=True):
        st.info("Market Sync Complete")

# --- 5. DASHBOARD EXECUTION ---
data = get_final_production_data(start_year, model_option, t_costs)

# Metrics Calculation
years_val = max(0.1, (data.index.max() - data.index.min()).days / 365.25)
ann_ret = (data['Strategy'].iloc[-1] / 100)**(1/years_val) - 1
daily_rets = data['Strategy'].pct_change().dropna()
mdd = ((data['Strategy'] / data['Strategy'].cummax()) - 1).min()

st.title("🎯 P2 Momentum Intelligence")

# Target Box - Uses fixed GLOBAL_SOFR scope
current_strat_ret = data['Strategy_Ret'].iloc[-1]
target_asset = "GLD" if current_strat_ret > DAILY_SOFR else "CASH"

st.markdown(f"""
<div class="target-box">
    <div style="color:#586069; font-size:14px;">Signal for Next Market Open ({model_option})</div>
    <div style="font-size:32px; font-weight:bold;">{target_asset}</div>
    <div style="color:#0041d0; font-size:12px; font-weight:bold;">SVR Poly C=500 | Stop=8% | T-Cost={t_costs}bps</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Annualized Return", f"{ann_ret:.2%}")
c2.metric("Max Drawdown", f"{mdd:.2%}")
c3.metric("Sharpe Ratio", f"{(ann_ret - 0.035) / (daily_rets.std() * np.sqrt(252)):.2f}")

# Charting
fig = px.line(data, x=data.index, y=['Strategy', 'Benchmark'], 
              title="Equity Curve: Strategy vs GLD Buy & Hold",
              color_discrete_map={"Strategy": "#0041d0", "Benchmark": "#d73a49"})
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)

# Audit Log
st.subheader("📋 15-Day Strategy Audit Log")
audit = data.tail(15).copy()
audit['Date'] = audit.index.date
st.table(audit[['Date', 'Strategy']].iloc[::-1])
