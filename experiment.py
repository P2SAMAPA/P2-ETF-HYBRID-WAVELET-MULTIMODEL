import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# --- ISOLATED UI CONFIG ---
st.set_page_config(layout="wide", page_title="P2 Lab: Kernel Experiments")

st.title("🧪 The Strategy Sandbox")
st.markdown("""
This file is isolated from `app.py`. We are testing three specific improvements:
1. **Standardization (Scaling)**: Ensuring all macro features have equal weight.
2. **Adaptive RBF**: High Gamma to focus on recent market regimes (2024-2026).
3. **Momentum Poly**: High C to aggressively chase price curves.
""")

# --- MOCK DATA LOADER (Wired to mimic your master_data structure) ---
@st.cache_data(ttl=3600)
def load_sandbox_data():
    # In production, this would be: df = pd.read_parquet('your_hf_link')
    dates = pd.date_range(start="2012-01-01", end=datetime.now() - timedelta(days=1), freq='B')
    df = pd.DataFrame(index=dates)
    
    # Simulating features that need scaling (VIX vs SOFR vs Returns)
    df['VIX'] = np.random.uniform(12, 35, len(dates)) # Large scale
    df['DXY'] = np.random.uniform(90, 110, len(dates)) # Medium scale
    df['SOFR'] = 0.0532 # Small scale
    
    # Target Assets
    df['GLD_Ret'] = np.random.normal(0.0005, 0.012, len(dates))
    df['SPY_Ret'] = np.random.normal(0.0004, 0.015, len(dates))
    df['CASH_Ret'] = 0.0532 / 360 # Your corrected SOFR math
    return df

df_raw = load_sandbox_data()

# --- THE EXPERIMENT ENGINE ---
def run_experiment(df, kernel_type, C_val, gamma_val, use_scaling=True):
    # 1. Feature Selection
    features = ['VIX', 'DXY', 'SOFR']
    X = df[features].values
    y = df['GLD_Ret'].values
    
    # 2. STEP A: Standardization Check
    if use_scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # 3. SVR Model Setup
    model = SVR(kernel=kernel_type, C=C_val, gamma=gamma_val)
    
    # Simulated Walk-forward (simplified for experiment speed)
    # In reality, this trains on historical and predicts next day
    model.fit(X[:1000], y[:1000]) 
    preds = model.predict(X)
    
    # 4. Strategy Mapping
    # If prediction > threshold, pick GLD, else CASH
    strat_rets = np.where(preds > 0, df['GLD_Ret'], df['CASH_Ret'])
    
    return (1 + strat_rets).cumprod() * 100

# --- UI CONTROLS ---
with st.sidebar:
    st.header("🔬 Experiment Parameters")
    start_yr = st.slider("Test Start Year", 2012, 2026, 2017)
    
    st.divider()
    st.subheader("Scenario A: Adaptive RBF")
    rbf_gamma = st.slider("RBF Gamma (Sensitivity)", 0.01, 1.0, 0.5)
    
    st.subheader("Scenario B: Momentum Poly")
    poly_c = st.slider("Poly C (Aggression)", 1, 200, 100)
    poly_deg = st.number_input("Poly Degree", 2, 5, 3)

# --- EXECUTION ---
# Baseline (Your Stable Logic - Unscaled RBF)
baseline_curve = run_experiment(df_raw, 'rbf', 1.0, 'scale', use_scaling=False)

# Challenger 1: Scaled + High Gamma RBF
adaptive_curve = run_experiment(df_raw, 'rbf', 1.0, rbf_gamma, use_scaling=True)

# Challenger 2: Scaled + High C Poly
momentum_curve = run_experiment(df_raw, 'poly', poly_c, 'scale', use_scaling=True)

# Filter for chosen period
results = pd.DataFrame({
    'Stable Baseline': baseline_curve,
    'Adaptive RBF (Scaled)': adaptive_curve,
    'Momentum Poly (Scaled)': momentum_curve
}, index=df_raw.index)

results_filtered = results[results.index.year >= start_yr]

# --- RESULTS DISPLAY ---
c1, c2, c3 = st.columns(3)

def get_metrics(series):
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    daily_rets = series.pct_change().dropna()
    mdd = ((series / series.cummax()) - 1).min()
    return f"{total_ret:.2%}", f"{mdd:.2%}"

# Metric Cards
m1_ret, m1_dd = get_metrics(results_filtered['Stable Baseline'])
c1.metric("Stable Baseline", m1_ret, f"DD: {m1_dd}", delta_color="inverse")

m2_ret, m2_dd = get_metrics(results_filtered['Adaptive RBF (Scaled)'])
c2.metric("Adaptive RBF", m2_ret, f"DD: {m2_dd}", delta_color="normal")

m3_ret, m3_dd = get_metrics(results_filtered['Momentum Poly (Scaled)'])
c3.metric("Momentum Poly", m3_ret, f"DD: {m3_dd}", delta_color="normal")

# Charting
st.divider()
fig = px.line(results_filtered, title="Champion vs. Challengers: Equity Curve Comparison")
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)

# --- COMPARISON TABLE ---
st.subheader("📊 Key Performance Differentiators")
comparison_data = {
    "Metric": ["Total Return", "Max Drawdown", "Volatility (Std)"],
    "Stable": [m1_ret, m1_dd, f"{results_filtered['Stable Baseline'].pct_change().std():.4%}"],
    "Adaptive RBF": [m2_ret, m2_dd, f"{results_filtered['Adaptive RBF (Scaled)'].pct_change().std():.4%}"],
    "Momentum Poly": [m3_ret, m3_dd, f"{results_filtered['Momentum Poly (Scaled)'].pct_change().std():.4%}"]
}
st.table(pd.DataFrame(comparison_data))

st.info("💡 **Insight:** Look at the 2024-2026 section of the graph. If Adaptive RBF is 'hugging' the price moves better than Baseline, the StandardScaler + High Gamma is working.")
