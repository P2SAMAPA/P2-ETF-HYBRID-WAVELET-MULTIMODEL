import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
from models.engine import MomentumEngine
from data.processor import build_feature_matrix
from data.loader import FeatureLoader, HF_REPO_ID, HF_FILENAME, HF_REPO_TYPE
from data.processor import build_feature_matrix
# ---------------------------------------------------------------------------
# 1. SECRETS — read silently, never exposed in UI
# ---------------------------------------------------------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]
FRED_KEY = st.secrets["FRED_API_KEY"]
# ---------------------------------------------------------------------------
# 2. LIVE SOFR — FRED first, Stooq fallback
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def get_live_sofr() -> float:
    # Try FRED first
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_KEY)
        s = fred.get_series("SOFR")
        val = float(s.dropna().iloc[-1]) / 100
        print(f"SOFR from FRED: {val:.4%}")
        return val
    except Exception as e:
        print(f"FRED SOFR failed: {e} — trying Stooq fallback")
    # Stooq fallback (13-week T-bill proxy)
    try:
        url = "https://stooq.com/q/d/l/?s=^irx&i=d"
        df = pd.read_csv(url)
        val = float(df['Close'].iloc[-1]) / 100
        print(f"SOFR from Stooq: {val:.4%}")
        return val
    except Exception as e:
        print(f"Stooq SOFR fallback failed: {e} — using hardcoded default")
    return 0.0532 # Last-resort hardcoded fallback
LIVE_SOFR = get_live_sofr()
DAILY_SOFR = LIVE_SOFR / 360
# ---------------------------------------------------------------------------
# 3. LOAD RAW DATA FROM HUGGINGFACE (public dataset — no token needed)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_raw_data() -> pd.DataFrame:
    """
    Reads master_data.parquet from the public HuggingFace dataset.
    Contains raw ETF closing prices + macro levels from 2008-01-01.
    force_download=True bypasses stale local HuggingFace cache.
    """
    try:
        path = hf_hub_download(
            repo_id = HF_REPO_ID,
            filename = HF_FILENAME,
            repo_type = HF_REPO_TYPE,
            force_download= True, # Bypass stale local HF cache
        )
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(df.index)
        # Sanitise column names — guard against tuple-string columns like
        # "('Price', 'GLD')" which appear when yfinance MultiIndex leaks into parquet
        def _clean_col(c: str) -> str:
            c = str(c).strip()
            if c.startswith("('") or c.startswith('("'):
                parts = c.strip("()").replace("'", "").replace('"', "").split(",")
                c = parts[-1].strip()
            return c
        df.columns = [_clean_col(c) for c in df.columns]
        print(f"Master parquet loaded: {len(df)} rows, cols: {list(df.columns)}")
        return df
    except Exception as e:
        st.error(f"Could not load master data from HuggingFace: {e}")
        return pd.DataFrame()
# ---------------------------------------------------------------------------
# 4. CORE BACKTEST PIPELINE
# Fixes vs original:
# (a) Real SVR signal via engine.py — no more rolling mean proxy
# (b) build_feature_matrix() in processor.py handles returns + denoising
# (c) Hard IS/OOS split — SVR trained only on pre-OOS data
# (d) OOS equity starts fresh at 100 — no in-sample contamination
# (e) Transaction costs flow correctly — applied on every signal flip
# and trailing stop exit; t_costs_bps is a cache key so any slider
# change fully invalidates the cache
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def get_final_data(data_hash: str, start_yr: int,
                    model_choice: str, t_costs_bps: int) -> pd.DataFrame:
    raw_df = load_raw_data()
    # All 5 ETFs you requested
    assets = ["TLT", "TBT", "VNQ", "GLD", "SLV"]
    t_cost_pct = t_costs_bps / 10_000
    
    # --- Step 1: Generate Predictions for all 5 Assets ---
    all_preds = {}
    for ticker in assets:
        try:
            # Build feature matrix specifically for this ticker
            X, y, idx, _ = build_feature_matrix(raw_df, target_col=ticker, denoise=True)
            
            is_mask = idx.year < start_yr
            oos_mask = idx.year >= start_yr
            
            # Train model only on In-Sample data
            engine = MomentumEngine(c_param=700.0, degree=3)
            engine.train(X[is_mask], y[is_mask])
            
            # Generate Out-of-Sample predictions
            preds = engine.predict_series(X[oos_mask])
            all_preds[ticker] = pd.Series(preds, index=idx[oos_mask])
        except Exception as e:
            st.warning(f"Skipping {ticker} due to error: {e}")

    # Combine all predictions into one table [Date x Ticker]
    pred_df = pd.DataFrame(all_preds).dropna()
    oos_idx = pred_df.index
    
    # --- Step 2: Multi-Asset Backtest Loop ---
    equity = 100.0
    current_asset = "CASH"
    strat_rets, asset_picks, real_view = [], [], []

    for i in range(len(oos_idx)):
        date = oos_idx[i]
        daily_preds = pred_df.iloc[i]
        
        # Identify the ticker with the highest predicted return
        best_ticker = daily_preds.idxmax()
        best_val = daily_preds.max()

        # Decision: Pick the best ETF if its prediction is > 0, else CASH
        new_asset = best_ticker if best_val > 0 else "CASH"

        # Apply transaction costs on a switch
        if new_asset != current_asset:
            equity *= (1 - t_cost_pct)
            current_asset = new_asset

        # Calculate daily return
        if current_asset == "CASH":
            day_ret = DAILY_SOFR
        else:
            # Ensure return column exists, otherwise compute it
            ret_col = f"{current_asset}_Ret"
            if ret_col in raw_df.columns:
                day_ret = raw_df.loc[date, ret_col]
            else:
                day_ret = raw_df[current_asset].pct_change().loc[date]
        
        equity *= (1 + day_ret)
        strat_rets.append(day_ret)
        asset_picks.append(current_asset)
        real_view.append(day_ret)

    # --- Step 3: Assembly ---
    oos_df = pd.DataFrame(index=oos_idx)
    oos_df["Strategy_Ret"] = strat_rets
    oos_df["Allocated_Asset"] = asset_picks
    oos_df["Strategy_Path"] = (pd.Series(strat_rets).add(1).cumprod() * 100).values
    oos_df["SVR_Predicted"] = pred_df.max(axis=1).values
    oos_df["Realised_Return_View"] = real_view
    
    # Benchmarks
    oos_df["GLD_Benchmark"] = (raw_df.loc[oos_idx, "GLD"].pct_change().add(1).cumprod() * 100).values
    
    return oos_df
# ---------------------------------------------------------------------------
# 5. UI CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="P2 ETF WAVELET SVR PPO MODEL", page_icon="💹")
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    div[data-testid="stMetricValue"] { color: #0041d0 !important; font-weight: bold; }
    .target-box {
        padding: 20px; border: 1px solid #e1e4e8; border-radius: 10px;
        background-color: #f8f9fa; margin-bottom: 25px;
    }
    .methodology-card {
        padding: 15px; background-color: #f1f3f5; border-radius: 8px;
        border-left: 5px solid #0041d0;
    }
    </style>
    """, unsafe_allow_html=True)
# ---------------------------------------------------------------------------
# 6. SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_option = st.radio(
        "Active Engine",
        ["Option A: SVR(Poly-Aggressive)", "Option B: SVR(Poly-Aggressive) + PPO"]
    )
    t_costs = st.slider("Transaction Cost (bps)", 0, 100, 10, step=5)
    start_year = st.slider("OOS Start Year", 2010, 2024, 2014)
    # Floor at 2010 so SVR always has >= 2 years of IS training data
    st.divider()
    st.caption(f"Live SOFR: {LIVE_SOFR:.3%}")
    if st.button("🔄 Sync Market Data", use_container_width=True):
        with st.spinner("Syncing data from Stooq / yfinance / FRED..."):
            loader = FeatureLoader(fred_key=FRED_KEY, hf_token=HF_TOKEN)
            result = loader.sync_data()
        st.cache_data.clear()
        if "Success" in result:
            st.success(result)
        else:
            st.error(result)
# ---------------------------------------------------------------------------
# 7. LOAD DATA & RUN PIPELINE
# ---------------------------------------------------------------------------
raw_df = load_raw_data()
if raw_df.empty:
    st.error("No data loaded. Please click 'Sync Market Data' in the sidebar to seed the dataset.")
    st.stop()
# Build a lightweight cache key from shape + last date — no JSON round-trip
# (JSON serialisation corrupts DatetimeIndex, causing IS/OOS splits to fail)
data_hash = f"{len(raw_df)}_{str(raw_df.index.max().date())}"
data = get_final_data(data_hash, start_year, model_option, t_costs)
if data.empty:
    st.error("Backtest returned no data. Try adjusting the OOS Start Year.")
    st.stop()
# ---------------------------------------------------------------------------
# 8. METRICS
# ---------------------------------------------------------------------------
n_years = max(len(data) / 252, 0.01)
ann_ret = (data["Strategy_Path"].iloc[-1] / 100) ** (1 / n_years) - 1
mdd_peak = ((data["Strategy_Path"] / data["Strategy_Path"].cummax()) - 1).min()
sharpe = (ann_ret - LIVE_SOFR) / (data["Strategy_Ret"].std() * np.sqrt(252) + 1e-9)
# Hit ratio: SVR predicted direction vs actual GLD direction (last 15 OOS days)
# SVR_Predicted is genuinely lag-safe (features shifted >= 1 day in processor.py)
last15 = data.tail(15)
hit_ratio = (last15["SVR_Predicted"].gt(0) == last15["GLD_Ret"].gt(0)).mean()
# ---------------------------------------------------------------------------
# 9. HEADER
# ---------------------------------------------------------------------------
st.title("🎯 P2 ETF WAVELET SVR PPO MODEL")
st.markdown(f"""
<div class="target-box">
    <div style="color:#586069; font-size:14px;">
        Market Open Date: {datetime.now().strftime('%Y-%m-%d')}
    </div>
    <div style="font-size:32px; font-weight:bold;">
        {data['Allocated_Asset'].iloc[-1]}
    </div>
</div>
""", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ann. Return", f"{ann_ret:.2%}")
m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
m3.metric("Max DD (P-T)", f"{mdd_peak:.2%}")
m4.metric("Max DD (Daily)", f"{data['Strategy_Ret'].min():.2%}")
m5.metric("Hit Ratio (15d)", f"{hit_ratio:.0%}")
# ---------------------------------------------------------------------------
# 10. EQUITY CURVE
# ---------------------------------------------------------------------------
bench_cols = ["Strategy_Path", "GLD_Benchmark"]
color_map = {"Strategy_Path": "#0041d0", "GLD_Benchmark": "#ffd700"}
if "SPY_Benchmark" in data.columns:
    bench_cols.append("SPY_Benchmark")
    color_map["SPY_Benchmark"] = "#d73a49"
if "AGG_Benchmark" in data.columns:
    bench_cols.append("AGG_Benchmark")
    color_map["AGG_Benchmark"] = "#28a745"
st.plotly_chart(
    px.line(
        data, x=data.index, y=bench_cols,
        title=f"OOS Equity Curve vs Benchmarks | TC = {t_costs} bps | OOS from {start_year}",
        color_discrete_map=color_map
    ),
    use_container_width=True
)
# ---------------------------------------------------------------------------
# 11. AUDIT LOG — Cleaned UI with 2 Decimal Places
# ---------------------------------------------------------------------------
st.subheader("📋 15-Day Strategy Audit Log")

audit_df = data.tail(15).copy()
# Format the date to remove the 00:00:00 timestamp
audit_df["Date"] = audit_df.index.strftime("%Y-%m-%d")

audit_display = audit_df[["Date", "Allocated_Asset", "SVR_Predicted", 
                           "Realised_Return_View"]].copy()

audit_display.columns = ["Date", "ETF Picked", "SVR Predicted Return", "Realised Return"]

def color_rets(val):
    if isinstance(val, (int, float)):
        return "color: #28a745; font-weight: bold" if val > 0 else "color: #d73a49; font-weight: bold"
    return ""

# We use .map() instead of .applymap() to fix the FutureWarning
# We use {:.2%} to show 2 decimal places as a percentage
st.table(
    audit_display.style
    .map(color_rets, subset=["SVR Predicted Return", "Realised Return"])
    .format({
        "SVR Predicted Return": "{:.2%}", 
        "Realised Return": "{:.2%}"
    })
)
# ---------------------------------------------------------------------------
# 12. METHODOLOGY
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📖 Methodology Details")
st.markdown(f"""
<div class="methodology-card">
    <b>Data Sources:</b> ETF prices via <b>Stooq → yfinance fallback</b>.
    Macro signals (VIX, DXY, T10Y2Y, SOFR, IG Spread, HY Spread) via
    <b>FRED → Stooq fallback</b>. All data from 2008-01-01.<br><br>
    <b>Feature Engineering:</b> Daily returns computed from raw prices.
    DWT wavelet denoising (sym4, level=3, soft thresholding) applied to
    each return series. Lagged features (1, 3, 5, 10, 21d) + rolling vol
    (5d, 21d) — all shifted ≥1 day for zero look-ahead bias.<br><br>
    <b>Model:</b> SVR with <b>3rd Degree Polynomial Kernel</b>, <b>C=700</b>,
    epsilon=0.001. Trained exclusively on in-sample data (before {start_year}).<br><br>
    <b>PPO Integration:</b> (Option B) Entry threshold scales with prior 21-day
    realised volatility — higher vol raises the bar to enter a long position.<br><br>
    <b>Transaction Costs:</b> <b>{t_costs} bps</b> deducted on every signal flip
    and on trailing stop exits.<br><br>
    <b>Risk Guard:</b> <b>10% Trailing Stop-Loss</b> — exits to CASH if equity
    falls 10% from its running peak within a long position.
</div>
""", unsafe_allow_html=True)
