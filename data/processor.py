import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# MACRO COLUMNS
# ---------------------------------------------------------------------------
MACRO_COLS_BASE = [
    "VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD",
    "DGS1MO", "DGS3MO", "DGS6MO",
    "DGS1", "DGS2", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30",
]

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

# Backward-compatible alias
MACRO_COLS = MACRO_COLS_BASE


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
# FIX: pct_change(fill_method='pad') is deprecated in pandas 2.x and will be
# removed. Use explicit ffill() first then pct_change(fill_method=None).
# ---------------------------------------------------------------------------
def _pct_change(series: pd.Series) -> pd.Series:
    """Forward-fill then compute pct_change with no implicit fill."""
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

    FIX (main bug): feat.dropna() was dropping ALL rows because options
    columns (SPY_IV_ATM_30D etc) are entirely NaN for historical data —
    options are live-only and only populate from the first daily_update run.
    Fixed by:
      1. Splitting features into CORE (always present) and OPTIONAL
         (ADV/options — may be NaN for historical rows).
      2. Calling dropna() only on CORE columns.
      3. Keeping optional feature NaNs as 0.0 (mean-imputed after scaling)
         so they contribute no signal until real data arrives.

    Returns
    -------
    X_scaled      : np.ndarray (n_samples, n_features)
    y             : np.ndarray (n_samples,)
    index         : pd.DatetimeIndex
    feature_names : list[str]
    """
    if target_col not in raw_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in raw data")

    # ── 1. Return series for feature symbols ─────────────────────────────
    ret_df = pd.DataFrame(index=raw_df.index)
    for sym in feature_symbols:
        if sym in raw_df.columns:
            # FIX Bug 2: use explicit ffill + fill_method=None
            ret_df[f"{sym}_Ret"] = _pct_change(raw_df[sym])
        else:
            ret_df[f"{sym}_Ret"] = np.nan

    # ── 2. Macro level features ───────────────────────────────────────────
    for col in MACRO_COLS_BASE:
        if col in raw_df.columns:
            ret_df[f"{col}_lvl"] = raw_df[col]

    # ── 3. DWT denoising ─────────────────────────────────────────────────
    if denoise:
        for col in ret_df.columns:
            if "_Ret" in col:
                ret_df[col] = apply_dwt_denoise(ret_df[col])

    # ── 4. ADV / volume features (log-scaled) ────────────────────────────
    opt_ret_cols = []   # track optional feature columns

    if include_volume:
        for adv_col in _get_adv_cols(raw_df, target_col):
            series = raw_df[adv_col].replace(0, np.nan)
            new_col = f"{adv_col}_log"
            ret_df[new_col] = np.log1p(series)
            opt_ret_cols.append(new_col)
        if market_proxy != target_col:
            for adv_col in _get_adv_cols(raw_df, market_proxy):
                series = raw_df[adv_col].replace(0, np.nan)
                new_col = f"MKT_{adv_col}_log"
                ret_df[new_col] = np.log1p(series)
                opt_ret_cols.append(new_col)

    # ── 5. Options scalar features ────────────────────────────────────────
    if include_options:
        for opt_col in _get_options_cols(raw_df, target_col):
            ret_df[opt_col] = raw_df[opt_col]
            opt_ret_cols.append(opt_col)
        if market_proxy != target_col:
            for opt_col in _get_options_cols(raw_df, market_proxy):
                mkt_col = f"MKT_{opt_col}"
                ret_df[mkt_col] = raw_df[opt_col]
                opt_ret_cols.append(mkt_col)
        spy_iv_col = f"{market_proxy}_IV_ATM_30D"
        if "VIX" in raw_df.columns and spy_iv_col in raw_df.columns:
            ret_df["VIX_IV_BASIS"] = raw_df["VIX"] / 100.0 - raw_df[spy_iv_col]
            opt_ret_cols.append("VIX_IV_BASIS")

    # ── 6. Build lagged feature matrix ────────────────────────────────────
    feat = pd.DataFrame(index=ret_df.index)

    # Core features: target return lags
    for lag in [1, 3, 5, 10, 21]:
        feat[f"target_lag{lag}"] = ret_df[f"{target_col}_Ret"].shift(lag)

    # Core features: lagged macro levels
    for col in MACRO_COLS_BASE:
        lvl_col = f"{col}_lvl"
        if lvl_col in ret_df.columns:
            feat[f"{col}_lag1"] = ret_df[lvl_col].shift(1)

    # Core features: cross-asset returns
    for sym in feature_symbols:
        if sym == target_col:
            continue
        feat[f"{sym}_ret"] = ret_df[f"{sym}_Ret"]

    # Optional features: ADV (lagged 1 day)
    optional_feat_cols = []
    if include_volume:
        for col in [c for c in ret_df.columns
                    if ("_ADV_" in c or "_DVOL_" in c) and "_log" in c]:
            lag_col = f"{col}_lag1"
            feat[lag_col] = ret_df[col].shift(1)
            optional_feat_cols.append(lag_col)

    # Optional features: options (lagged 1 day)
    if include_options:
        for col in [c for c in ret_df.columns
                    if any(s in c for s in OPTIONS_SIGNAL_SUFFIXES)
                    or c == "VIX_IV_BASIS"]:
            lag_col = f"{col}_lag1"
            feat[lag_col] = ret_df[col].shift(1)
            optional_feat_cols.append(lag_col)

    # ── 7. Target variable ────────────────────────────────────────────────
    # FIX Bug 2: use explicit ffill + fill_method=None
    feat["__target__"] = _pct_change(raw_df[target_col])

    # ── 8. Drop NaNs — CORE COLUMNS ONLY ─────────────────────────────────
    # FIX Bug 1: original feat.dropna() dropped all rows because optional
    # columns (options signals) are entirely NaN for historical data.
    # We only require the core columns to be non-NaN.
    core_feat_cols = [c for c in feat.columns
                      if c not in optional_feat_cols and c != "__target__"]
    core_feat_cols_present = [c for c in core_feat_cols if c in feat.columns]

    feat = feat.dropna(subset=core_feat_cols_present + ["__target__"]).iloc[22:]

    # Fill remaining NaNs in optional columns with 0 (mean-imputed equivalent
    # after StandardScaler — contributes no signal until real data arrives)
    if optional_feat_cols:
        present_opt = [c for c in optional_feat_cols if c in feat.columns]
        feat[present_opt] = feat[present_opt].fillna(0.0)

    if len(feat) == 0:
        raise ValueError(
            f"Feature matrix is empty for target '{target_col}' after dropna "
            f"on core columns. Check that price and macro data are available."
        )

    feature_names = [c for c in feat.columns if c != "__target__"]

    # ── 9. Scale features ─────────────────────────────────────────────────
    X_raw    = feat[feature_names].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, feat["__target__"].values, feat.index, feature_names
