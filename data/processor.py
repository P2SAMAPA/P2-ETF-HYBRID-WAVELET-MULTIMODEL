import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# MACRO COLUMNS
# Updated to include all treasury yield columns added to master data,
# plus new options-derived scalar signals and ADV volume signals.
# ---------------------------------------------------------------------------

# Base macro columns (rates, spreads, FX, vol)
MACRO_COLS_BASE = [
    "VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD",
    "DGS1MO", "DGS3MO", "DGS6MO",
    "DGS1", "DGS2", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30",
]

# Options scalar signal column suffixes — resolved dynamically per ticker
# These are added as features when available in the DataFrame.
# Pattern: <TICKER>_<SIGNAL>  e.g. SPY_IV_ATM_30D, SPY_SKEW_30D
OPTIONS_SIGNAL_SUFFIXES = [
    "IV_ATM_30D",
    "IV_ATM_60D",
    "IV_ATM_90D",
    "SKEW_30D",
    "IV_TERM_SLOPE",
    "IV_SURFACE_LEVEL",
    "PCR_OI_30D",
    "PCR_VOL_30D",
    "IMPLIED_MOVE_7D",
    "IMPLIED_MOVE_30D",
    "VRP_30D",
    "GEX",
]

# ADV / dollar volume column suffixes — resolved dynamically per ticker
# Pattern: <TICKER>_ADV_<W>D or <TICKER>_DVOL_<W>D
ADV_SIGNAL_SUFFIXES = [
    "ADV_20D",
    "ADV_63D",
    "ADV_252D",
    "DVOL_20D",
    "DVOL_63D",
    "DVOL_252D",
]

# Combined macro columns list for backward compatibility
MACRO_COLS = MACRO_COLS_BASE


def _get_options_cols(raw_df: pd.DataFrame, ticker: str) -> list:
    """
    Return options signal column names for a given ticker that actually
    exist in raw_df. Avoids KeyErrors if options data not yet populated.
    """
    return [
        f"{ticker}_{suffix}"
        for suffix in OPTIONS_SIGNAL_SUFFIXES
        if f"{ticker}_{suffix}" in raw_df.columns
    ]


def _get_adv_cols(raw_df: pd.DataFrame, ticker: str) -> list:
    """
    Return ADV / dollar volume column names for a given ticker that exist
    in raw_df.
    """
    return [
        f"{ticker}_{suffix}"
        for suffix in ADV_SIGNAL_SUFFIXES
        if f"{ticker}_{suffix}" in raw_df.columns
    ]


# ---------------------------------------------------------------------------
# WAVELET DENOISING (unchanged)
# ---------------------------------------------------------------------------
def apply_dwt_denoise(series: pd.Series,
                       wavelet: str = "sym4",
                       level: int = 3) -> pd.Series:
    clean = series.dropna()
    if len(clean) < (2 ** level):
        return series
    if clean.std() == 0:
        return series
    nan_mask = series.isna()
    data     = series.fillna(0.0).values.astype(np.float64)
    coeffs   = pywt.wavedec(data, wavelet, level=level)
    sigma    = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    if threshold > 0:
        denoised_coeffs = [coeffs[0]] + [
            pywt.threshold(c, value=threshold, mode="soft")
            for c in coeffs[1:]
        ]
        reconstructed = pywt.waverec(denoised_coeffs, wavelet)[:len(data)]
    else:
        reconstructed = data
    result           = pd.Series(reconstructed, index=series.index, name=series.name)
    result[nan_mask] = np.nan
    return result


# ---------------------------------------------------------------------------
# FEATURE MATRIX BUILDER
# Updated to incorporate:
#   1. ADV / dollar volume features (log-scaled, lagged)
#   2. Options scalar signals (IV, skew, PCR, implied move, VRP, GEX)
#   3. Cross-asset options features from related tickers (e.g. SPY options
#      as market-wide signal when building features for any equity ETF)
# ---------------------------------------------------------------------------
def build_feature_matrix(
    raw_df: pd.DataFrame,
    target_col: str,
    feature_symbols: list,
    denoise: bool = True,
    include_options: bool = True,
    include_volume: bool = True,
    market_proxy: str = "SPY",
) -> tuple:
    """
    Build feature matrix for a given target asset.

    Parameters
    ----------
    raw_df          : master DataFrame containing prices, macro, ADV, options cols
    target_col      : ticker of the asset being predicted
    feature_symbols : list of tickers in the same universe (cross-asset features)
    denoise         : apply DWT denoising to return series
    include_options : add options-derived scalar features (IV, skew, PCR etc)
    include_volume  : add ADV / dollar volume features
    market_proxy    : ticker to use for market-wide options features (default SPY)

    Returns
    -------
    X_scaled        : np.ndarray (n_samples, n_features) — standardised features
    y               : np.ndarray (n_samples,) — target next-day returns
    index           : pd.DatetimeIndex of sample dates
    feature_names   : list of feature column names
    """
    if target_col not in raw_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in raw data")

    # ── 1. Return series for all feature symbols ──────────────────────────
    ret_df = pd.DataFrame(index=raw_df.index)
    for sym in feature_symbols:
        if sym in raw_df.columns:
            ret_df[f"{sym}_Ret"] = raw_df[sym].pct_change()
        else:
            ret_df[f"{sym}_Ret"] = np.nan

    # ── 2. Macro level features ───────────────────────────────────────────
    for col in MACRO_COLS_BASE:
        if col in raw_df.columns:
            ret_df[f"{col}_lvl"] = raw_df[col]

    # ── 3. DWT denoising on return columns ────────────────────────────────
    if denoise:
        for col in ret_df.columns:
            if "_Ret" in col:
                ret_df[col] = apply_dwt_denoise(ret_df[col])

    # ── 4. ADV / volume features (log-scaled) ────────────────────────────
    if include_volume:
        # Target ticker's own volume features
        for adv_col in _get_adv_cols(raw_df, target_col):
            series = raw_df[adv_col].replace(0, np.nan)
            # Log-scale dollar volume (spans many orders of magnitude)
            if "DVOL" in adv_col:
                ret_df[f"{adv_col}_log"] = np.log1p(series)
            else:
                ret_df[f"{adv_col}_log"] = np.log1p(series)
        # Market proxy volume (SPY ADV as liquidity regime indicator)
        if market_proxy != target_col:
            for adv_col in _get_adv_cols(raw_df, market_proxy):
                series = raw_df[adv_col].replace(0, np.nan)
                ret_df[f"MKT_{adv_col}_log"] = np.log1p(series)

    # ── 5. Options scalar features ────────────────────────────────────────
    if include_options:
        # Target ticker's own options signals (if liquid options exist)
        for opt_col in _get_options_cols(raw_df, target_col):
            ret_df[opt_col] = raw_df[opt_col]
        # Market-wide options signals from SPY (always informative regardless of target)
        if market_proxy != target_col:
            for opt_col in _get_options_cols(raw_df, market_proxy):
                ret_df[f"MKT_{opt_col}"] = raw_df[opt_col]
        # VIX-IV divergence: VIX minus SPY IV_ATM_30D (basis signal)
        spy_iv_col = f"{market_proxy}_IV_ATM_30D"
        if "VIX" in raw_df.columns and spy_iv_col in raw_df.columns:
            ret_df["VIX_IV_BASIS"] = raw_df["VIX"] / 100.0 - raw_df[spy_iv_col]

    # ── 6. Build lagged feature matrix ────────────────────────────────────
    feat = pd.DataFrame(index=ret_df.index)

    # Lags of target returns (momentum/reversal signal)
    for lag in [1, 3, 5, 10, 21]:
        feat[f"target_lag{lag}"] = ret_df[f"{target_col}_Ret"].shift(lag)

    # Lagged macro levels
    for col in MACRO_COLS_BASE:
        lvl_col = f"{col}_lvl"
        if lvl_col in ret_df.columns:
            feat[f"{col}_lag1"] = ret_df[lvl_col].shift(1)

    # Cross-asset returns (contemporaneous — other ETFs as factor proxies)
    for sym in feature_symbols:
        if sym == target_col:
            continue
        feat[f"{sym}_ret"] = ret_df[f"{sym}_Ret"]

    # ADV features (lagged 1 day — can't use today's volume for prediction)
    if include_volume:
        vol_cols = [c for c in ret_df.columns
                    if ("_ADV_" in c or "_DVOL_" in c) and "_log" in c]
        for col in vol_cols:
            feat[f"{col}_lag1"] = ret_df[col].shift(1)

    # Options features (lagged 1 day — use yesterday's options signal)
    if include_options:
        opt_cols = [c for c in ret_df.columns
                    if any(suf in c for suf in OPTIONS_SIGNAL_SUFFIXES)
                    or c == "VIX_IV_BASIS"]
        for col in opt_cols:
            feat[f"{col}_lag1"] = ret_df[col].shift(1)

    # ── 7. Target variable ────────────────────────────────────────────────
    feat["__target__"] = raw_df[target_col].pct_change()

    # ── 8. Drop NaNs and burn-in period ──────────────────────────────────
    feat         = feat.dropna().iloc[22:]
    feature_names = [c for c in feat.columns if c != "__target__"]

    # ── 9. Scale features ─────────────────────────────────────────────────
    X_raw    = feat[feature_names].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, feat["__target__"].values, feat.index, feature_names
