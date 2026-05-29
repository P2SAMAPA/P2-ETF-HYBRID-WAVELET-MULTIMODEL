import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# MACRO COLUMNS
# ---------------------------------------------------------------------------
MACRO_COLS_CORE = [
    "VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD",
]

MACRO_COLS_EXTENDED = [
    "DGS1MO", "DGS3MO", "DGS6MO",
    "DGS1", "DGS2", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30",
]

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
# CANONICAL FEATURE NAME LIST
# Defines the FIXED set of feature columns every ETF will have.
# Built once from the full universe — ensures identical width for all assets.
# ---------------------------------------------------------------------------
def get_canonical_feature_names(
    raw_df: pd.DataFrame,
    feature_symbols: list,
    include_options: bool = True,
    include_volume: bool = True,
    market_proxy: str = "SPY",
) -> list:
    """
    Return the ordered list of feature column names that build_feature_matrix
    will produce for ANY ticker in feature_symbols.

    Using a canonical list guarantees identical feature width across all
    ETFs even when some have shorter histories (e.g. VCIT starts 2009).
    """
    names = []

    # Core: target lags
    for lag in [1, 3, 5, 10, 21]:
        names.append(f"target_lag{lag}")

    # Core: CORE macro lags
    for col in MACRO_COLS_CORE:
        if col in raw_df.columns:
            names.append(f"{col}_lag1")

    # Core: cross-asset returns (all symbols except target — target varies,
    # so we list all; the caller passes the same feature_symbols for all ETFs)
    for sym in feature_symbols:
        names.append(f"{sym}_ret")   # target's own slot is overwritten per call

    # Optional: EXTENDED macro lags
    for col in MACRO_COLS_EXTENDED:
        if col in raw_df.columns:
            names.append(f"{col}_lag1")

    # Optional: ADV columns for market_proxy (always present for all ETFs)
    if include_volume:
        for adv_col in _get_adv_cols(raw_df, market_proxy):
            names.append(f"MKT_{adv_col}_log")

    # Optional: options columns for market_proxy
    if include_options:
        for opt_col in _get_options_cols(raw_df, market_proxy):
            names.append(f"MKT_{opt_col}")
        spy_iv_col = f"{market_proxy}_IV_ATM_30D"
        if "VIX" in raw_df.columns and spy_iv_col in raw_df.columns:
            names.append("VIX_IV_BASIS")

    # Deduplicate while preserving order
    seen, result = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


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


def _pct_change(series: pd.Series) -> pd.Series:
    """Fix for pandas 2.x FutureWarning: explicit ffill before pct_change."""
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
    canonical_names: list = None,
) -> tuple:
    """
    Build feature matrix for a given target asset.

    KEY FIX — consistent feature width across all ETFs
    ---------------------------------------------------
    The original code produced different numbers of feature columns per ETF
    (e.g. 63 for TLT vs 51 for VCIT) because:
      1. MACRO_COLS_EXTENDED (DGS series) included DGS20 which only starts 2022,
         causing dropna() to cut rows to ~741.
      2. Per-ticker optional columns (ADV, options for each ticker) varied by
         availability, creating different column counts.

    Fix:
      - CORE columns (MACRO_COLS_CORE, target lags, cross-asset) → dropna() enforced
      - EXTENDED/optional columns → filled with 0.0 when NaN, never cause row drops
      - Per-ticker ADV/options columns are EXCLUDED from the canonical feature set
        (only market_proxy ADV/options are included — same for all ETFs)
      - If canonical_names is provided, the output is aligned to that exact column
        list, guaranteeing identical width for concatenation in train_models.py

    Returns
    -------
    X_scaled      : np.ndarray (n_samples, n_features)
    y             : np.ndarray (n_samples,)
    index         : pd.DatetimeIndex
    feature_names : list[str]
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

    # ── 3. DWT denoising ─────────────────────────────────────────────────
    if denoise:
        for col in ret_df.columns:
            if "_Ret" in col:
                ret_df[col] = apply_dwt_denoise(ret_df[col])

    # ── 4. Market-proxy ADV features (same for all ETFs) ──────────────────
    if include_volume and market_proxy != target_col:
        for adv_col in _get_adv_cols(raw_df, market_proxy):
            series  = raw_df[adv_col].replace(0, np.nan)
            ret_df[f"MKT_{adv_col}_log"] = np.log1p(series)

    # ── 5. Market-proxy options features (same for all ETFs) ──────────────
    if include_options:
        if market_proxy != target_col:
            for opt_col in _get_options_cols(raw_df, market_proxy):
                ret_df[f"MKT_{opt_col}"] = raw_df[opt_col]
        spy_iv_col = f"{market_proxy}_IV_ATM_30D"
        if "VIX" in raw_df.columns and spy_iv_col in raw_df.columns:
            ret_df["VIX_IV_BASIS"] = raw_df["VIX"] / 100.0 - raw_df[spy_iv_col]

    # ── 6. Build lagged feature matrix ────────────────────────────────────
    feat      = pd.DataFrame(index=ret_df.index)
    core_cols = []

    # CORE: target return lags
    for lag in [1, 3, 5, 10, 21]:
        col = f"target_lag{lag}"
        feat[col] = ret_df[f"{target_col}_Ret"].shift(lag)
        core_cols.append(col)

    # CORE: CORE macro lags (full history from 2008)
    for col in MACRO_COLS_CORE:
        if f"{col}_lvl" in ret_df.columns:
            lag_col = f"{col}_lag1"
            feat[lag_col] = ret_df[f"{col}_lvl"].shift(1)
            core_cols.append(lag_col)

    # CORE: cross-asset returns
    for sym in feature_symbols:
        if sym == target_col:
            continue
        col = f"{sym}_ret"
        feat[col] = ret_df[f"{sym}_Ret"]
        core_cols.append(col)

    # OPTIONAL: EXTENDED macro lags (some start late — filled 0 if NaN)
    for col in MACRO_COLS_EXTENDED:
        if f"{col}_lvl" in ret_df.columns:
            lag_col = f"{col}_lag1"
            feat[lag_col] = ret_df[f"{col}_lvl"].shift(1)

    # OPTIONAL: market-proxy ADV lags
    if include_volume:
        for col in [c for c in ret_df.columns
                    if c.startswith("MKT_") and "_log" in c]:
            feat[f"{col}_lag1"] = ret_df[col].shift(1)

    # OPTIONAL: market-proxy options lags
    if include_options:
        for col in [c for c in ret_df.columns
                    if (c.startswith("MKT_") and
                        any(s in c for s in OPTIONS_SIGNAL_SUFFIXES))
                    or c == "VIX_IV_BASIS"]:
            feat[f"{col}_lag1"] = ret_df[col].shift(1)

    # ── 7. Target variable ────────────────────────────────────────────────
    feat["__target__"] = _pct_change(raw_df[target_col])

    # ── 8. Drop NaNs on CORE columns only + burn-in ───────────────────────
    core_present = [c for c in core_cols if c in feat.columns]
    feat = feat.dropna(subset=core_present + ["__target__"]).iloc[22:]

    # Fill all remaining NaNs with 0 (optional cols — no signal until live)
    feat = feat.fillna(0.0)

    if len(feat) == 0:
        raise ValueError(
            f"Feature matrix empty for '{target_col}'. "
            "Check price and core macro data availability."
        )

    feature_names = [c for c in feat.columns if c != "__target__"]

    # ── 9. Align to canonical feature list (guarantees consistent width) ──
    # This is the key fix for the concatenation error in train_models.py.
    # Every ETF's feature matrix is padded/trimmed to the same column list.
    if canonical_names is not None:
        # Add any missing canonical columns as zeros
        for cn in canonical_names:
            if cn not in feat.columns:
                feat[cn] = 0.0
        # Select exactly the canonical columns in canonical order
        # (drops per-ticker optional cols not in canonical list)
        feat_aligned   = feat[canonical_names]
        feature_names  = canonical_names
        X_raw          = feat_aligned.values
    else:
        X_raw = feat[feature_names].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, feat["__target__"].values, feat.index, feature_names
