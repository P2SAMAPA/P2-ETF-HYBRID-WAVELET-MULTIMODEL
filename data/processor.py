import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# MACRO COLUMNS
# ---------------------------------------------------------------------------

# CORE macro columns — available from 2008 in master data.
# These are used in dropna() so they must have full history.
MACRO_COLS_CORE = [
    "VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD",
]

# EXTENDED macro columns — treasury yields added in June 2026 seeding.
# DGS series from FRED have varying start dates:
#   DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS7, DGS10 → available from ~2008
#   DGS20 → only available from April 2022 (FRED discontinued 2001-2010, resumed 2022)
#   DGS30 → available from ~2008 but with gaps
# Treated as OPTIONAL — included as features when available but NOT in dropna() subset.
MACRO_COLS_EXTENDED = [
    "DGS1MO", "DGS3MO", "DGS6MO",
    "DGS1", "DGS2", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30",
]

# Combined list — kept for backward compatibility with any code importing MACRO_COLS
MACRO_COLS_BASE = MACRO_COLS_CORE + MACRO_COLS_EXTENDED
MACRO_COLS      = MACRO_COLS_BASE   # backward-compatible alias

OPTIONS_SIGNAL_SUFFIXES = [
    "IV_ATM_30D", "IV_ATM_60D", "IV_ATM_90D",
    "SKEW_30D", "IV_TERM_SLOPE", "IV_SURFACE_LEVEL",
    "PCR_OI_30D", "PCR_VOL_30D",
    "IMPLIED_MOVE_7D", "IMPLIED_MOVE_30D",
    "VRP_30D", "GEX",
]

ADV_SIGNAL_SUFFIXES = [
    "ADV_20D", "ADV_63D", "ADV_252D",
    "DVOL_20D", "DVOL_63D", "DVOL_252D",
]


def _get_options_cols(raw_df: pd.DataFrame, ticker: str) -> list:
    return [f"{ticker}_{s}" for s in OPTIONS_SIGNAL_SUFFIXES
            if f"{ticker}_{s}" in raw_df.columns]


def _get_adv_cols(raw_df: pd.DataFrame, ticker: str) -> list:
    return [f"{ticker}_{s}" for s in ADV_SIGNAL_SUFFIXES
            if f"{ticker}_{s}" in raw_df.columns]


# ---------------------------------------------------------------------------
# WAVELET DENOISING
# ---------------------------------------------------------------------------
def apply_dwt_denoise(series: pd.Series,
                       wavelet: str = "sym4",
                       level: int = 3) -> pd.Series:
    clean = series.dropna()
    if len(clean) < (2 ** level):
        return series
    if clean.std() == 0:
        return series
    nan_mask  = series.isna()
    data      = series.fillna(0.0).values.astype(np.float64)
    coeffs    = pywt.wavedec(data, wavelet, level=level)
    sigma     = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    if threshold > 0:
        denoised = [coeffs[0]] + [
            pywt.threshold(c, value=threshold, mode="soft") for c in coeffs[1:]
        ]
        reconstructed = pywt.waverec(denoised, wavelet)[:len(data)]
    else:
        reconstructed = data
    result           = pd.Series(reconstructed, index=series.index, name=series.name)
    result[nan_mask] = np.nan
    return result


# ---------------------------------------------------------------------------
# SAFE PCT_CHANGE
# pct_change(fill_method='pad') deprecated in pandas 2.x — use explicit ffill
# ---------------------------------------------------------------------------
def _pct_change(series: pd.Series) -> pd.Series:
    return series.ffill().pct_change(fill_method=None)


# ---------------------------------------------------------------------------
# FEATURE MATRIX BUILDER
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

    Feature classification
    ----------------------
    CORE features — dropna() enforced, full history from 2008:
      - Target return lags (1,3,5,10,21 days)
      - MACRO_COLS_CORE lags: VIX, DXY, T10Y2Y, IG_SPREAD, HY_SPREAD
      - Cross-asset returns (other ETFs in universe)

    OPTIONAL features — filled with 0.0 if NaN, NOT in dropna() subset:
      - MACRO_COLS_EXTENDED lags: DGS series (some start later, e.g. DGS20 from 2022)
      - ADV / dollar volume lags (full history from 2008 seeding)
      - Options signal lags (live-only, NaN for historical rows)

    This ensures ~4750 training samples from 2008 rather than ~741 from 2022.
    """
    if target_col not in raw_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in raw data")

    # ── 1. Return series ──────────────────────────────────────────────────
    ret_df = pd.DataFrame(index=raw_df.index)
    for sym in feature_symbols:
        if sym in raw_df.columns:
            ret_df[f"{sym}_Ret"] = _pct_change(raw_df[sym])
        else:
            ret_df[f"{sym}_Ret"] = np.nan

    # ── 2. All macro level features ───────────────────────────────────────
    for col in MACRO_COLS_BASE:
        if col in raw_df.columns:
            ret_df[f"{col}_lvl"] = raw_df[col]

    # ── 3. DWT denoising on return columns ────────────────────────────────
    if denoise:
        for col in ret_df.columns:
            if "_Ret" in col:
                ret_df[col] = apply_dwt_denoise(ret_df[col])

    # ── 4. ADV / volume features (log-scaled, optional) ───────────────────
    optional_ret_cols = []
    if include_volume:
        for adv_col in _get_adv_cols(raw_df, target_col):
            series  = raw_df[adv_col].replace(0, np.nan)
            new_col = f"{adv_col}_log"
            ret_df[new_col] = np.log1p(series)
            optional_ret_cols.append(new_col)
        if market_proxy != target_col:
            for adv_col in _get_adv_cols(raw_df, market_proxy):
                series  = raw_df[adv_col].replace(0, np.nan)
                new_col = f"MKT_{adv_col}_log"
                ret_df[new_col] = np.log1p(series)
                optional_ret_cols.append(new_col)

    # ── 5. Options scalar features (optional) ────────────────────────────
    if include_options:
        for opt_col in _get_options_cols(raw_df, target_col):
            ret_df[opt_col] = raw_df[opt_col]
            optional_ret_cols.append(opt_col)
        if market_proxy != target_col:
            for opt_col in _get_options_cols(raw_df, market_proxy):
                mkt_col = f"MKT_{opt_col}"
                ret_df[mkt_col] = raw_df[opt_col]
                optional_ret_cols.append(mkt_col)
        spy_iv_col = f"{market_proxy}_IV_ATM_30D"
        if "VIX" in raw_df.columns and spy_iv_col in raw_df.columns:
            ret_df["VIX_IV_BASIS"] = raw_df["VIX"] / 100.0 - raw_df[spy_iv_col]
            optional_ret_cols.append("VIX_IV_BASIS")

    # ── 6. Build lagged feature matrix ────────────────────────────────────
    feat         = pd.DataFrame(index=ret_df.index)
    core_cols    = []   # must be non-NaN → in dropna() subset
    optional_cols= []   # may be NaN for some historical rows → filled with 0

    # CORE: target return lags
    for lag in [1, 3, 5, 10, 21]:
        col = f"target_lag{lag}"
        feat[col] = ret_df[f"{target_col}_Ret"].shift(lag)
        core_cols.append(col)

    # CORE: lagged CORE macro levels (VIX, DXY, T10Y2Y, IG_SPREAD, HY_SPREAD)
    for col in MACRO_COLS_CORE:
        lvl_col = f"{col}_lvl"
        if lvl_col in ret_df.columns:
            lag_col = f"{col}_lag1"
            feat[lag_col] = ret_df[lvl_col].shift(1)
            core_cols.append(lag_col)

    # CORE: cross-asset returns
    for sym in feature_symbols:
        if sym == target_col:
            continue
        col = f"{sym}_ret"
        feat[col] = ret_df[f"{sym}_Ret"]
        core_cols.append(col)

    # OPTIONAL: lagged EXTENDED macro levels (DGS series — some start late)
    for col in MACRO_COLS_EXTENDED:
        lvl_col = f"{col}_lvl"
        if lvl_col in ret_df.columns:
            lag_col = f"{col}_lag1"
            feat[lag_col] = ret_df[lvl_col].shift(1)
            optional_cols.append(lag_col)

    # OPTIONAL: ADV features (lagged 1 day)
    if include_volume:
        for col in [c for c in ret_df.columns
                    if ("_ADV_" in c or "_DVOL_" in c) and "_log" in c]:
            lag_col = f"{col}_lag1"
            feat[lag_col] = ret_df[col].shift(1)
            optional_cols.append(lag_col)

    # OPTIONAL: options features (lagged 1 day)
    if include_options:
        for col in [c for c in ret_df.columns
                    if any(s in c for s in OPTIONS_SIGNAL_SUFFIXES)
                    or c == "VIX_IV_BASIS"]:
            lag_col = f"{col}_lag1"
            feat[lag_col] = ret_df[col].shift(1)
            optional_cols.append(lag_col)

    # ── 7. Target variable ────────────────────────────────────────────────
    feat["__target__"] = _pct_change(raw_df[target_col])

    # ── 8. Drop NaNs on CORE columns only + burn-in ───────────────────────
    core_present = [c for c in core_cols if c in feat.columns]
    feat = feat.dropna(subset=core_present + ["__target__"]).iloc[22:]

    # Fill optional column NaNs with 0 (mean-imputed after StandardScaler)
    opt_present = [c for c in optional_cols if c in feat.columns]
    if opt_present:
        feat[opt_present] = feat[opt_present].fillna(0.0)

    if len(feat) == 0:
        raise ValueError(
            f"Feature matrix empty for '{target_col}' after core dropna. "
            f"Check price and macro data availability."
        )

    feature_names = [c for c in feat.columns if c != "__target__"]
    X_raw    = feat[feature_names].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, feat["__target__"].values, feat.index, feature_names
