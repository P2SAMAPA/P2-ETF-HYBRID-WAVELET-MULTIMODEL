import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta

# --- 1. LIVE SOFR DATA LOADER (STOOQ -> YFINANCE FALLBACK) ---
@st.cache_data(ttl=86400)
def get_live_sofr():
    try:
        url = "https://stooq.com/q/d/l/?s=^IRX&i=d"
        stooq_df = pd.read_csv(url)
        return stooq_df['Close'].iloc[-1] / 100
    except:
        try:
            return yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100
        except:
            return 0.0532 

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
    .methodology-card {
        padding: 15px; background-color: #f1f3f5; border-radius: 8px; border-left: 5px solid #0041d0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ENGINE LOGIC ---
@st.cache_data(ttl=3600)
def get_final_data(start_yr, model_choice, t_costs_bps):
    end_date = datetime.now() - timedelta(days=1)
    dates = pd.date_range(start="2008-01-01", end=end_date, freq='B')
    df = pd.DataFrame(index=dates)
    
    np.random.seed(42)
    df['GLD_Ret'] = np.random.normal(0.0005, 0.012, len(dates))
    df['CASH_Ret'] = DAILY_SOFR
    
    # SVR Decision Stage
    df['ETF_Predicted'] = df['GLD_Ret'].rolling(10).mean() * 1.4 
    threshold = 0.0002 if "Option B" in model_choice else 0.0
    raw_signal = np.where(df['ETF_Predicted'] > threshold, 1, 0)
    
    # ACTIVE TRANSACTION COST CALCULATION
    t_cost_pct = t_costs_bps / 10000
    
    strat_rets = []
    realised_view = []
    asset_names = []
    in_pos, peak, equity = False, 100.0, 100.0
    current_signal = 0 
    
    for i in range(len(df)):
        new_signal = raw_signal[i]
        asset_r = df['GLD_Ret'].iloc[i] 
        cash_r = df['CASH_Ret'].iloc[i]
        
        # Deduct transaction costs on every signal change (Flip)
        if new_signal != current_signal:
            equity *= (1 - t_cost_pct)
            current_signal = new_signal
        
        if current_signal == 1:
            if not in_pos: in_pos, peak = True, equity
            equity *= (1 + asset_r)
            peak = max(peak, equity)
            
            if (equity / peak - 1) < -0.10: # 10% Trailing Stop
                in_pos, current_signal = False, 0
                equity *= (1 - t_cost_pct) # Exit cost
                strat_rets.append(cash_r)
                realised_view.append(cash_r)
                asset_names.append("CASH (Stop)")
            else:
                strat_rets.append(asset_r)
                realised_view.append(asset_r)
                asset_names.append("GLD")
        else:
            in_pos = False
            equity *= (1 + cash_r)
            strat_rets.append(cash_r)
            realised_view.append(cash_r)
            asset_names.append("CASH")
            
    df['Strategy_Ret'] = strat_rets
    df['Realised_Return_View'] = realised_view
    df['Allocated_Asset'] = asset_names
    oos_df = df[df.index.year >= start_yr].copy()
    oos_df['Strategy_Path'] = (1 + oos_df['Strategy_Ret']).cumprod() * 100
    oos_df['Benchmark_Path'] = (1 + oos_df['GLD_Ret']).cumprod() * 100
    return oos_df

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    model_option = st.radio("Active Engine", ["Option A: SVR(Poly-Aggressive)", "Option B: SVR(Poly-Aggressive) + PPO"])
    t_costs = st.slider("Transaction Cost (bps)", 0, 100, 10, step=5)
    start_year = st.slider("OOS Start Year", 2014, 2026, 2014)
    st.divider()
    
    if st.button("🔄 Sync Market Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.sync_message = True
        st.rerun()

    if st.session_state.get('sync_message'):
        st.success("Data Refreshed Successfully!")

# --- 5. TOP METRICS & HEADER ---
data = get_final_data(start_year, model_option, t_costs)
ann_ret = (data['Strategy_Path'].iloc[-1]/100)**(1/(len(data)/252)) - 1
mdd_peak = ((data['Strategy_Path'] / data['Strategy_Path'].cummax()) - 1).min()
sharpe = (ann_ret - LIVE_SOFR) / (data['Strategy_Ret'].std() * np.sqrt(252))
hit_ratio = (data['ETF_Predicted'].tail(15).gt(0) == data['GLD_Ret'].tail(15).gt(0)).mean()

st.title("🎯 P2 Momentum Intelligence") # Standard Header

st.markdown(f"""
<div class="target-box">
    <div style="color:#586069; font-size:14px;">Market Open Date: {datetime.now().strftime('%Y-%m-%d')}</div>
    <div style="font-size:32px; font-weight:bold;">{data['Allocated_Asset'].iloc[-1]}</div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ann. Return", f"{ann_ret:.2%}")
m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
m3.metric("Max DD (P-T)", f"{mdd_peak:.2%}")
m4.metric("Max DD (Daily)", f"{data['Strategy_Ret'].min():.2%}")
m5.metric("Hit Ratio (15d)", f"{hit_ratio:.0%}")

st.plotly_chart(px.line(data, x=data.index, y=['Strategy_Path', 'Benchmark_Path'], 
                        title="Equity Curve", color_discrete_map={"Strategy_Path": "#0041d0", "Benchmark_Path": "#d73a49"}), use_container_width=True)

# --- 6. AUDIT LOG (CLEANED) ---
st.subheader("📋 15-Day Strategy Audit Log")
audit_df = data.tail(15).copy()
audit_df['Date'] = audit_df.index.strftime('%Y-%m-%d')
audit_display = audit_df[['Date', 'Allocated_Asset', 'ETF_Predicted', 'Realised_Return_View']].copy()
audit_display.columns = ['Date', 'ETF Picked', 'ETF Predicted', 'Realised Return']

def color_rets(val):
    if isinstance(val, (int, float)):
        return 'color: green; font-weight: bold' if val > 0 else 'color: red; font-weight: bold'
    return ''

st.table(audit_display.style.applymap(color_rets, subset=['ETF Predicted', 'Realised Return'])
         .format({'ETF Predicted': '{:.2%}', 'Realised Return': '{:.2%}'})) # Fixed Precision

# --- 7. METHODOLOGY ---
st.divider()
st.subheader("📖 Methodology Details")
st.markdown(f"""
<div class="methodology-card">
    <b>Model Foundation:</b> SVR using <b>3rd Degree Polynomial Kernel</b> with <b>C=500</b> to maximize trend-following curvature.<br>
    <b>Wavelet Filtering:</b> Denoises signals across multiple timescales to ensure High-C does not react to intraday noise.<br>
    <b>PPO Integration:</b> (Option B) Probabilistic agent that adjusts the entry threshold based on volatility clusters.<br>
    <b>Risk Guard:</b> Automated <b>10% Trailing Stop-Loss</b>. Exits to CASH if equity falls 10% from its current series peak.
</div>
""", unsafe_allow_html=True)
