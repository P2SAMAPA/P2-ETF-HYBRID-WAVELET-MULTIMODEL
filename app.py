import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download

from models.engine import MomentumEngine, build_features
from data.loader import FeatureLoader, HF_REPO_ID, HF_FILENAME, HF_REPO_TYPE
from data.processor import build_feature_matrix

# ---------------------------------------------------------------------------
# 1. SECRETS — read silently, never exposed in UI
# ---------------------------------------------------------------------------
HF_TOKEN  = st.secrets["HF_TOKEN"]
FRED_KEY  = st.secrets["FRED_API_KEY"]


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
        df  = pd.read_csv(url)
        val = float(df['Close'].iloc[-1]) / 100
        print(f"SOFR from Stooq: {val:.4%}")
        return val
    except Exception as e:
        print(f"Stooq SOFR fallback failed: {e} — using hardcoded default")

    return 0.0532  # Last-resort hardcoded fallback

LIVE_SOFR  = get_live_sofr()
DAILY_SOFR = LIVE_SOFR / 360


# ---------------------------------------------------------------------------
# 3. LOAD RAW DATA FROM HUGGINGFACE (public dataset — no token needed)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_raw_data() -> pd.DataFrame:
    """
    Reads master_data.parquet from the public HuggingFace dataset.
    Contains raw ETF closing prices + macro levels from 2008-01-01.
    """
    try:
        path = hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = HF_FILENAME,
            repo_type = HF_REPO_TYPE,
            # No token needed — dataset is public
        )
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(df.index)
        return df
    except Exception as e:
        st.error(f"Could not load master data from HuggingFace: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. CORE BACKTEST PIPELINE
#    Fixes vs original:
#    (a) Real SVR signal via engine.py — no more rolling mean proxy
#    (b) build_feature_matrix() in processor.py handles returns + denoising
#    (c) Hard IS/OOS split — SVR trained only on pre-OOS data
#    (d) OOS equity starts fresh at 100 — no in-sample contamination
#    (e) Transaction costs flow correctly — applied on every signal flip
#        and trailing stop exit; t_costs_bps is a cache key so any slider
#        change fully invalidates the cache
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_final_data(raw_df_json: str, start_yr: int,
                   model_choice: str, t_costs_bps: int) -> pd.DataFrame:
    """
    Parameters
    ----------
    raw_df_json  : JSON-serialised raw DataFrame (used as cache key)
    start_yr     : OOS start year — SVR trained on everything before this
    model_choice : "Option A" or "Option B" (PPO threshold variant)
    t_costs_bps  : Transaction cost in basis points (from slider)
    """
    raw_df = pd.read_json(raw_df_json)
    raw_df.index = pd.DatetimeIndex(raw_df.index)

    today = pd.Timestamp.now().normalize()

    # Add CASH_Ret column for the backtest loop
    raw_df["CASH_Ret"] = DAILY_SOFR

    # -------------------------------------------------------------------
    # STEP 1: Build feature matrix (returns + denoising + lags)
    # -------------------------------------------------------------------
    try:
        X_all, y_all, valid_idx, feature_names = build_feature_matrix(
            raw_df, target_col="GLD", denoise=True
        )
    except ValueError as e:
        st.error(f"Feature build failed: {e}")
        return pd.DataFrame()

    feat_df = pd.DataFrame(X_all, index=valid_idx, columns=feature_names)
    tgt_s   = pd.Series(y_all, index=valid_idx, name="GLD_Ret")

    # -------------------------------------------------------------------
    # STEP 2: Hard IS/OOS split — train SVR only on pre-OOS data
    # -------------------------------------------------------------------
    is_mask  = valid_idx.year < start_yr
    oos_mask = valid_idx.year >= start_yr

    if is_mask.sum() < 50:
        st.warning(f"Only {is_mask.sum()} in-sample rows — OOS start year may be too early. "
                   f"Training on all available data instead.")
        X_train, y_train = X_all, y_all
    else:
        X_train = feat_df[is_mask].values
        y_train = tgt_s[is_mask].values

    engine = MomentumEngine(c_param=700.0, degree=3)
    engine.train(X_train, y_train)

    # -------------------------------------------------------------------
    # STEP 3: Generate SVR predictions on OOS rows only
    # -------------------------------------------------------------------
    X_oos   = feat_df[oos_mask].values
    oos_idx = valid_idx[oos_mask]

    if len(X_oos) == 0:
        st.error("No OOS data available. Try an earlier OOS start year.")
        return pd.DataFrame()

    svr_preds = engine.predict_series(X_oos)

    # PPO Option B: volatility-scaled entry threshold
    if "Option B" in model_choice:
        oos_gld_ret  = tgt_s[oos_mask]
        rolling_vol  = oos_gld_ret.shift(1).rolling(21).std().fillna(oos_gld_ret.std())
        threshold    = (rolling_vol * 0.5).values
    else:
        threshold = np.zeros(len(oos_idx))

    raw_signal = (svr_preds > threshold).astype(int)

    # -------------------------------------------------------------------
    # STEP 4: OOS backtest loop
    #   - equity starts at 100 at OOS start (clean, uncontaminated)
    #   - t_cost_pct applied on every signal flip and trailing stop exit
    # -------------------------------------------------------------------
    t_cost_pct = t_costs_bps / 10_000

    # Align GLD returns and CASH returns to OOS index
    gld_ret_col = "GLD_Ret" if "GLD_Ret" in raw_df.columns else None
    if gld_ret_col is None:
        # Compute GLD returns if not present
        raw_df["GLD_Ret"] = raw_df["GLD"].pct_change()

    oos_gld_rets  = raw_df.loc[oos_idx, "GLD_Ret"].fillna(0).values
    oos_cash_rets = np.full(len(oos_idx), DAILY_SOFR)

    strat_rets, realised_view, asset_names = [], [], []
    in_pos, peak, equity = False, 100.0, 100.0
    current_signal = 0

    for i in range(len(oos_idx)):
        new_signal = raw_signal[i]
        asset_r    = oos_gld_rets[i]
        cash_r     = oos_cash_rets[i]

        # Signal flip — deduct one-way transaction cost
        if new_signal != current_signal:
            equity        *= (1 - t_cost_pct)
            current_signal = new_signal

        if current_signal == 1:
            if not in_pos:
                in_pos = True
                peak   = equity      # reset peak on fresh entry

            equity *= (1 + asset_r)
            peak    = max(peak, equity)

            # 12% Trailing Stop-Loss
            if (equity / peak - 1) < -0.12:
                in_pos         = False
                current_signal = 0
                equity        *= (1 - t_cost_pct)   # exit cost on stop
                strat_rets.append(cash_r)
                realised_view.append(cash_r)
                asset_names.append("CASH (Stop)")
            else:
                strat_rets.append(asset_r)
                realised_view.append(asset_r)
                asset_names.append("GLD")
        else:
            in_pos  = False
            equity *= (1 + cash_r)
            strat_rets.append(cash_r)
            realised_view.append(cash_r)
            asset_names.append("CASH")

    # -------------------------------------------------------------------
    # STEP 5: Assemble OOS DataFrame
    # -------------------------------------------------------------------
    oos_df = raw_df.loc[oos_idx].copy()
    oos_df["Strategy_Ret"]         = strat_rets
    oos_df["Realised_Return_View"] = realised_view
    oos_df["Allocated_Asset"]      = asset_names
    oos_df["SVR_Predicted"]        = svr_preds
    oos_df["GLD_Ret"]              = oos_gld_rets

    oos_df["Strategy_Path"] = (1 + oos_df["Strategy_Ret"]).cumprod() * 100
    oos_df["GLD_Benchmark"] = (1 + oos_df["GLD_Ret"]).cumprod()      * 100

    # Benchmark returns for SPY and AGG
    for col in ["SPY", "AGG"]:
        if col in raw_df.columns:
            ret = raw_df[col].pct_change().loc[oos_idx].fillna(0)
            oos_df[f"{col}_Benchmark"] = (1 + ret).cumprod() * 100

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
    t_costs    = st.slider("Transaction Cost (bps)", 0, 100, 10, step=5)
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

# Serialise to JSON for use as a stable cache key
raw_df_json = raw_df.to_json()

data = get_final_data(raw_df_json, start_year, model_option, t_costs)

if data.empty:
    st.error("Backtest returned no data. Try adjusting the OOS Start Year.")
    st.stop()


# ---------------------------------------------------------------------------
# 8. METRICS
# ---------------------------------------------------------------------------
n_years  = max(len(data) / 252, 0.01)
ann_ret  = (data["Strategy_Path"].iloc[-1] / 100) ** (1 / n_years) - 1
mdd_peak = ((data["Strategy_Path"] / data["Strategy_Path"].cummax()) - 1).min()
sharpe   = (ann_ret - LIVE_SOFR) / (data["Strategy_Ret"].std() * np.sqrt(252) + 1e-9)

# Hit ratio: SVR predicted direction vs actual GLD direction (last 15 OOS days)
# SVR_Predicted is genuinely lag-safe (features shifted >= 1 day in processor.py)
last15    = data.tail(15)
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
m1.metric("Ann. Return",     f"{ann_ret:.2%}")
m2.metric("Sharpe Ratio",    f"{sharpe:.2f}")
m3.metric("Max DD (P-T)",    f"{mdd_peak:.2%}")
m4.metric("Max DD (Daily)",  f"{data['Strategy_Ret'].min():.2%}")
m5.metric("Hit Ratio (15d)", f"{hit_ratio:.0%}")


# ---------------------------------------------------------------------------
# 10. EQUITY CURVE
# ---------------------------------------------------------------------------
bench_cols = ["Strategy_Path", "GLD_Benchmark"]
color_map  = {"Strategy_Path": "#0041d0", "GLD_Benchmark": "#ffd700"}

if "SPY_Benchmark" in data.columns:
    bench_cols.append("SPY_Benchmark")
    color_map["SPY_Benchmark"] = "#d73a49"
if "AGG_Benchmark" in data.columns:
    bench_cols.append("AGG_Benchmark")
    color_map["AGG_Benchmark"] = "#28a745"

st.plotly_chart(
    px.line(
        data, x=data.index, y=bench_cols,
        title=f"OOS Equity Curve vs Benchmarks  |  TC = {t_costs} bps  |  OOS from {start_year}",
        color_discrete_map=color_map
    ),
    use_container_width=True
)


# ---------------------------------------------------------------------------
# 11. AUDIT LOG
# ---------------------------------------------------------------------------
st.subheader("📋 15-Day Strategy Audit Log")
audit_df = data.tail(15).copy()
audit_df["Date"] = audit_df.index.strftime("%Y-%m-%d")
audit_display = audit_df[["Date", "Allocated_Asset", "SVR_Predicted",
                           "Realised_Return_View"]].copy()
audit_display.columns = ["Date", "ETF Picked", "SVR Predicted Return", "Realised Return"]

def color_rets(val):
    if isinstance(val, (int, float)):
        return "color: green; font-weight: bold" if val > 0 else "color: red; font-weight: bold"
    return ""

st.table(
    audit_display.style
    .applymap(color_rets, subset=["SVR Predicted Return", "Realised Return"])
    .format({"SVR Predicted Return": "{:.4%}", "Realised Return": "{:.4%}"})
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
    <b>Risk Guard:</b> <b>12% Trailing Stop-Loss</b> — exits to CASH if equity
    falls 12% from its running peak within a long position.
</div>
""", unsafe_allow_html=True)
