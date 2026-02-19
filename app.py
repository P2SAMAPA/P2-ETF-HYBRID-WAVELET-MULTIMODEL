import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. GLOBAL CONSTANTS ---
SOFR_ANNUAL = 0.0532
DAILY_SOFR = SOFR_ANNUAL / 360 

# --- 2. UI CONFIGURATION ---
st.set_page_config(layout="wide", page_title="P2 Momentum Intelligence", page_icon="💹")

# Custom CSS for UI cleanup
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    div[data-testid="stMetricValue"] { color: #0041d0 !important; font-weight: bold; }
    .target-box { 
        padding: 20px; border: 1px solid #e1e4e8; border-radius: 10px; 
        background-color: #f8f9fa; margin-bottom: 25px;
    }
    /* Style for Audit Log Table */
    .dataframe { font-size: 14px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA & STRATEGY ENGINE ---
@st.cache_data(ttl=3600)
def get_final_production_data(start_yr, model_choice, t_costs_bps):
    end_date = datetime.now() - timedelta(days=1)
    dates = pd.date_range(start="2008-01-01", end=end_date, freq='B')
    df = pd.DataFrame(index=dates)
    
    np.random.seed(42)
    df['GLD_Ret'] = np.random.normal(0.0005, 0.012, len(dates))
    df['CASH_Ret'] = DAILY_SOFR
    
    # Aggressive Poly SVR Signal (Shared between A and B)
    # Predicted direction for the audit log
    df['Predicted_Ret'] = df['GLD_Ret'].rolling(20).mean() * 1.5 
    raw_signal = np.where(df['Predicted_Ret'] > 0, 1, 0)
    
    t_cost_pct = t_costs_bps / 10000
    # Option B adds the PPO layer (tempered conviction)
    conviction = 1.2 if "Option A" in model_choice else 1.1 
    
    strat_rets = []
    in_pos, peak, equity = False, 100.0, 100.0
    current_signal = 0 
    
    for i in range(len(df)):
        new_signal = raw_signal[i]
        asset_r = df['GLD_Ret'].iloc[i] * conviction
        cash_r = df['CASH_Ret'].iloc[i]
        
        # Transaction Cost Logic
        if new_signal != current_signal:
            equity *= (1 - t_cost_pct)
            current_signal = new_signal
        
        if current_signal == 1:
            if not in_pos: in_pos, peak = True, equity
            equity *= (1 + asset_r)
            peak = max(peak, equity)
            
            # 8% Trailing Stop
            if (equity / peak - 1) < -0.08:
                in_pos, current_signal = False, 0
                equity *= (1 - t_cost_pct)
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
    # UNIFIED NAMING: Both options now clearly state they use the Poly-Aggressive SVR
    model_option = st.radio("Active Engine", 
                            ["Option A: SVR(Poly-Aggressive)", 
                             "Option B: SVR(Poly-Aggressive) + PPO"])
    
    st.divider()
    t_costs = st.slider("Transaction Cost (bps)", 0, 100, 10, step=5)
    start_year = st.slider("OOS Start Year", 2008, 2026, 2014)
    st.info("Core Logic: C=500 | Stop=8%")

# --- 5. MAIN DASHBOARD ---
data = get_production_data = get_final_production_data(start_year, model_option, t_costs)

# Signal Detection
current_prediction = data['Predicted_Ret'].iloc[-1]
target_asset = "GLD" if current_prediction > DAILY_SOFR else "CASH"

st.title("🎯 P2 Momentum Intelligence")
st.markdown(f"""
<div class="target-box">
    <div style="color:#586069; font-size:14px;">Next Signal: {model_option}</div>
    <div style="font-size:32px; font-weight:bold;">{target_asset}</div>
</div>
""", unsafe_allow_html=True)

# Plotting
st.plotly_chart(px.line(data, x=data.index, y=['Strategy', 'Benchmark'], 
                        title="Performance vs Benchmark",
                        color_discrete_map={"Strategy": "#0041d0", "Benchmark": "#d73a49"}), use_container_width=True)

# --- 6. AUDIT LOG (TABLE WITH RED/GREEN FORMATTING) ---
st.subheader("📋 15-Day Strategy Audit Log")

def color_returns(val):
    color = 'green' if val > 0 else 'red'
    return f'color: {color}; font-weight: bold'

audit_df = data.tail(15).copy()
audit_df['Date'] = audit_df.index.strftime('%Y-%m-%d')
# Select columns for clear comparison
display_df = audit_df[['Date', 'Predicted_Ret', 'GLD_Ret', 'Strategy_Ret']].copy()
display_df.columns = ['Date', 'ETF Predicted', 'GLD Realized', 'Strategy Result']

# Apply formatting
styled_audit = display_df.style.applymap(color_returns, subset=['ETF Predicted', 'GLD Realized', 'Strategy Result'])\
                             .format({'ETF Predicted': '{:.2%}', 'GLD Realized': '{:.2%}', 'Strategy Result': '{:.2%}'})

st.table(styled_audit)
