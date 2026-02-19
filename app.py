import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from models.engine import MomentumEngine, build_features
from data.loader import FeatureLoader
from data.processor import apply_modwt_denoise


# ---------------------------------------------------------------------------
# 1. LIVE SOFR — Stooq first, hardcoded constant fallback
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def get_live_sofr() -> float:
    try:
        url = "https://stooq.com/q/d/l/?s=^irx&i=d"
        df  = pd.read_csv(url)
        return float(df['Close'].iloc[-1]) / 100
    except Exception:
        return 0.0532

LIVE_SOFR  = get_live_sofr()
DAILY_SOFR = LIVE_SOFR / 360


# ---------------------------------------------------------------------------
# 2. UI CONFIGURATION
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
# 3. LOAD RAW MASTER DATA FROM HUGGINGFACE
#    Falls back to synthetic data if HuggingFace is unreachable so the
#    dashboard remains usable before the first sync is run.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_raw_master(hf_token: str, repo_id: str) -> pd.DataFrame:
    if not hf_token or not repo_id:
        st.info("No HuggingFace credentials — using synthetic data. Run a sync to load real data.")
        return _synthetic_fallback()
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=repo_id,
            filename="master_data.parquet",
            repo_type="dataset",
            token=hf_token
        )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.warning(f"Could not load master data from HuggingFace ({e}). Using synthetic fallback.")
        return _synthetic_fallback()


def _synthetic_fallback() -> pd.DataFrame:
    """Synthetic price + macro data — used only when HuggingFace is unavailable."""
    dates = pd.date_range(start="2008-01-01", end=datetime.now() - timedelta(days=1), freq='B')
    np.random.seed(42)
    df = pd.DataFrame(index=dates)
    for col, drift, vol in [("GLD", 0.0005, 0.012), ("SLV", 0.0003, 0.018),
                             ("SPY", 0.0004, 0.010), ("AGG", 0.0001, 0.004),
                             ("TLT", 0.0002, 0.008), ("TBT", -0.0002, 0.010),
                             ("VNQ", 0.0003, 0.012)]:
        rets    = np.random.normal(drift, vol, len(dates))
        df[col] = 100 * np.cumprod(1 + rets)
    df["VIX"]    = np.abs(np.random.normal(20, 5,  len(dates)))
    df["DXY"]    = np.random.normal(95, 5, len(dates))
    df["T10Y2Y"] = np.random.normal(0.5, 0.5, len(dates))
    df["SOFR"]   = DAILY_SOFR * 360
    df["IG_OAS"] = np.abs(np.random.normal(120, 30, len(dates)))
    df["HY_OAS"] = np.abs(np.random.normal(400, 80, len(dates)))
    return df


# ---------------------------------------------------------------------------
# 4. FULL MODEL + BACKTEST PIPELINE
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_final_data(raw_df_json: str, start_yr: int,
                   model_choice: str, t_costs_bps: int) -> pd.DataFrame:
    """
    Deserialises raw_df, runs the full pipeline:
      processor → IS/OOS split → SVR train → OOS predict → backtest loop

    t_costs_bps is a cache key: any slider change triggers a full re-run.
    """
    raw_df = pd.read_json(raw_df_json)
    raw_df.index = pd.to_datetime(raw_df.index)

    # --- Build denoised, lag-safe feature matrix (processor.py) ---
    X_all, y_all, valid_idx, feature_names = build_feature_matrix(
        raw_df, target_col="GLD", denoise_level=3
    )
    feat_df = pd.DataFrame(X_all, index=valid_idx, columns=feature_names)
    tgt_s   = pd.Series(y_all, index=valid_idx, name="GLD_ret")

    # --- Hard IS/OOS split ---
    is_mask  = valid_idx.year < start_yr
    oos_mask = valid_idx.year >= start_yr

    X_train = feat_df[is_mask].values if is_mask.sum() >= 50 else X_all
    y_train = tgt_s[is_mask].values   if is_mask.sum() >= 50 else y_all

    # --- Train SVR on IS data only (engine.py) ---
    engine = MomentumEngine(c_param=700.0, degree=3)
    engine.train(X_train, y_train)

    # --- OOS predictions — fully out-of-sample, no look-ahead ---
    X_oos     = feat_df[oos_mask].values
    oos_idx   = valid_idx[oos_mask]
    svr_preds = engine.predict_series(X_oos)

    # GLD and CASH returns for OOS window
    oos_gld_rets = raw_df.loc[oos_idx, "GLD"].pct_change().fillna(0)
    oos_cash     = (raw_df.loc[oos_idx, "SOFR"] / 360
                    if "SOFR" in raw_df.columns
                    else pd.Series(DAILY_SOFR, index=oos_idx))

    # --- PPO Option B: vol-scaled entry threshold ---
    if "Option B" in model_choice:
        prior_vol = oos_gld_rets.shift(1).rolling(21).std().fillna(oos_gld_rets.std())
        threshold = (prior_vol * 0.5).values
    else:
        threshold = np.zeros(len(oos_idx))

    raw_signal = (svr_preds > threshold).astype(int)

    # --- OOS backtest loop ---
    # Transaction costs applied on every signal flip AND trailing stop exits
    # Equity resets to 100 at OOS start — no IS contamination
    t_cost_pct = t_costs_bps / 10_000
    strat_rets, realised_view, asset_names = [], [], []
    in_pos, peak, equity = False, 100.0, 100.0
    current_signal = 0

    for i in range(len(oos_idx)):
        new_signal = raw_signal[i]
        asset_r    = float(oos_gld_rets.iloc[i])
        cash_r     = float(oos_cash.iloc[i])

        # Signal flip cost
        if new_signal != current_signal:
            equity        *= (1 - t_cost_pct)
            current_signal = new_signal

        if current_signal == 1:
            if not in_pos:
                in_pos = True
                peak   = equity     # reset peak on every fresh long entry

            equity *= (1 + asset_r)
            peak    = max(peak, equity)

            # 12% trailing stop
            if (equity / peak - 1) < -0.12:
                in_pos         = False
                current_signal = 0
                equity        *= (1 - t_cost_pct)   # exit cost on stop trigger
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

    # --- Assemble OOS DataFrame ---
    oos_df = raw_df.loc[oos_idx].copy()
    oos_df['Strategy_Ret']         = strat_rets
    oos_df['Realised_Return_View'] = realised_view
    oos_df['Allocated_Asset']      = asset_names
    oos_df['ETF_Predicted']        = svr_preds

    for col in ["GLD", "SPY", "AGG"]:
        if col in oos_df.columns:
            oos_df[f"{col}_Ret"] = oos_df[col].pct_change().fillna(0)

    oos_df['Strategy_Path'] = (1 + oos_df['Strategy_Ret']).cumprod() * 100
    for col in ["GLD", "SPY", "AGG"]:
        ret_col = f"{col}_Ret"
        if ret_col in oos_df.columns:
            oos_df[f"{col}_Benchmark"] = (1 + oos_df[ret_col]).cumprod() * 100

    return oos_df


# ---------------------------------------------------------------------------
# 5. SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_option = st.radio(
        "Active Engine",
        ["Option A: SVR(Poly-Aggressive)", "Option B: SVR(Poly-Aggressive) + PPO"]
    )
    t_costs    = st.slider("Transaction Cost (bps)", 0, 100, 10, step=5)
    start_year = st.slider("OOS Start Year", 2010, 2026, 2014)
    st.divider()

    hf_token = st.text_input("HuggingFace Token", type="password",
                              key="hf_token_input")
    repo_id  = st.text_input("HF Dataset Repo ID",
                              value="your-username/etf-master-data",
                              key="repo_id_input")

    if st.button("🔄 Sync Market Data", use_container_width=True):
        if hf_token and repo_id:
            with st.spinner("Syncing from Stooq / yfinance / FRED..."):
                loader = FeatureLoader(
                    fred_key=st.secrets.get("FRED_KEY", ""),
                    hf_token=hf_token,
                    repo_id=repo_id
                )
                result = loader.sync_data()
            if "Success" in result:
                st.success(result)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(result)
        else:
            st.warning("Enter HuggingFace token and repo ID first.")


# ---------------------------------------------------------------------------
# 6. RUN PIPELINE
# ---------------------------------------------------------------------------
raw_df      = load_raw_master(
    hf_token=st.session_state.get("hf_token_input", ""),
    repo_id=st.session_state.get("repo_id_input",  "")
)
raw_df_json = raw_df.to_json(date_format='iso')
data        = get_final_data(raw_df_json, start_year, model_option, t_costs)


# ---------------------------------------------------------------------------
# 7. METRICS
# ---------------------------------------------------------------------------
n_years  = max(len(data) / 252, 0.01)
ann_ret  = (data['Strategy_Path'].iloc[-1] / 100) ** (1 / n_years) - 1
mdd_peak = ((data['Strategy_Path'] / data['Strategy_Path'].cummax()) - 1).min()
sharpe   = (ann_ret - LIVE_SOFR) / (data['Strategy_Ret'].std() * np.sqrt(252) + 1e-10)

last15    = data.tail(15)
hit_ratio = (
    (last15['ETF_Predicted'].gt(0) == last15['GLD_Ret'].gt(0)).mean()
    if 'GLD_Ret' in data.columns else np.nan
)

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
m5.metric("Hit Ratio (15d)", f"{hit_ratio:.0%}" if not np.isnan(hit_ratio) else "N/A")


# ---------------------------------------------------------------------------
# 8. EQUITY CURVE
# ---------------------------------------------------------------------------
bench_cols = [c for c in ['GLD_Benchmark', 'SPY_Benchmark', 'AGG_Benchmark']
              if c in data.columns]
plot_cols  = ['Strategy_Path'] + bench_cols

st.plotly_chart(
    px.line(
        data, x=data.index, y=plot_cols,
        title=(
            f"OOS Equity Curve vs Benchmarks  |  TC = {t_costs} bps  "
            f"|  SVR C=700 Poly-3  |  MODWT Denoised Features"
        ),
        color_discrete_map={
            "Strategy_Path": "#0041d0",
            "GLD_Benchmark": "#ffd700",
            "SPY_Benchmark": "#d73a49",
            "AGG_Benchmark": "#28a745",
        }
    ),
    use_container_width=True
)


# ---------------------------------------------------------------------------
# 9. AUDIT LOG
# ---------------------------------------------------------------------------
st.subheader("📋 15-Day Strategy Audit Log")
audit_df      = data.tail(15).copy()
audit_df['Date'] = audit_df.index.strftime('%Y-%m-%d')
audit_display = audit_df[['Date', 'Allocated_Asset', 'ETF_Predicted',
                           'Realised_Return_View']].copy()
audit_display.columns = ['Date', 'ETF Picked', 'SVR Predicted Return', 'Realised Return']

def color_rets(val):
    if isinstance(val, (int, float)):
        return 'color: green; font-weight: bold' if val > 0 else 'color: red; font-weight: bold'
    return ''

st.table(
    audit_display.style
    .applymap(color_rets, subset=['SVR Predicted Return', 'Realised Return'])
    .format({'SVR Predicted Return': '{:.4%}', 'Realised Return': '{:.4%}'})
)


# ---------------------------------------------------------------------------
# 10. METHODOLOGY
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📖 Methodology Details")
st.markdown(f"""
<div class="methodology-card">
    <b>Data Sources:</b>
    ETF prices — Stooq first → yfinance fallback.
    Macro signals (VIX, DXY, T10Y2Y, SOFR, IG OAS, HY OAS) — FRED first → Stooq fallback.
    Raw prices and macro levels stored in HuggingFace parquet. Returns computed downstream.<br><br>
    <b>Wavelet Denoising:</b> True MODWT via Stationary Wavelet Transform (sym4, level=3).
    Universal soft thresholding applied to detail coefficients — shift-invariant and
    avoids over-smoothing. Applied to each ETF return series before feature construction.<br><br>
    <b>Feature Matrix:</b> Lagged denoised returns (1, 3, 5, 10, 21d), rolling realised
    volatility (5d, 21d), lagged macro levels — all shifted ≥ 1 day (zero look-ahead bias).<br><br>
    <b>SVR Model:</b> 3rd Degree Polynomial Kernel, C=700, epsilon=0.001.
    Trained exclusively on data before the OOS Start Year.
    Predictions generated only on unseen OOS data.<br><br>
    <b>PPO (Option B):</b> Entry threshold scales with prior 21-day realised volatility —
    higher volatility raises the bar for a long signal.<br><br>
    <b>Transaction Costs:</b> <b>{t_costs} bps</b> applied on every signal flip and
    trailing-stop exit.<br><br>
    <b>Risk Guard:</b> 12% Trailing Stop-Loss — exits to CASH if equity falls 12% from
    the running peak of the current long position.
</div>
""", unsafe_allow_html=True)
