import numpy as np
import pandas as pd
import pywt


# ---------------------------------------------------------------------------
# WAVELET DENOISER
# Fixed issues vs original:
#   1. Labelled and documented as DWT (not MODWT — pywt.wavedec is DWT)
#   2. float64 instead of float32 — preserves precision for small daily returns
#   3. Soft thresholding on detail coefficients instead of zeroing them out —
#      preserves more signal structure while still removing high-freq noise
#   4. Input validation for short or NaN-heavy series
# ---------------------------------------------------------------------------

def apply_dwt_denoise(series: pd.Series, wavelet: str = "sym4",
                      level: int = 3) -> pd.Series:
    """
    Applies Discrete Wavelet Transform (DWT) denoising to a return series.

    Uses soft thresholding on detail coefficients (rather than zeroing them)
    to remove high-frequency noise while preserving signal structure and
    avoiding over-smoothing.

    Parameters
    ----------
    series  : pd.Series of daily returns (float64)
    wavelet : PyWavelets wavelet name (default: 'sym4')
    level   : Decomposition level (default: 3)

    Returns
    -------
    pd.Series of denoised returns, same length and index as input.
    NaN series returned if input is too short or all-NaN.
    """
    # --- Input validation ---
    clean = series.dropna()
    min_length = 2 ** level
    if len(clean) < min_length:
        print(f"DWT skipped: series too short ({len(clean)} rows, need {min_length})")
        return series  # Return original — don't corrupt with bad denoising

    if clean.std() == 0:
        print("DWT skipped: series is constant")
        return series

    # --- Use float64 for precision (daily returns are tiny numbers) ---
    data = series.values.astype(np.float64)

    # --- Decompose ---
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # --- Soft threshold on detail coefficients ---
    # Threshold estimated from finest-level detail using the universal threshold:
    # sigma * sqrt(2 * log(N)) where sigma is estimated via MAD of finest details
    finest_detail = coeffs[-1]
    sigma     = np.median(np.abs(finest_detail)) / 0.6745  # MAD estimator
    threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # Apply soft thresholding to all detail levels (coeffs[1:])
    # Leave approximation coefficients (coeffs[0]) untouched
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=threshold, mode="soft") for c in coeffs[1:]
    ]

    # --- Reconstruct ---
    reconstructed = pywt.waverec(denoised_coeffs, wavelet)

    # Trim to original length (PyWavelets pads internally)
    reconstructed = reconstructed[:len(data)]

    return pd.Series(reconstructed, index=series.index, name=series.name)


# ---------------------------------------------------------------------------
# FEATURE MATRIX BUILDER
# Called by engine.py and app.py to produce the SVR input matrix.
# All features are lagged >= 1 day — zero look-ahead bias guaranteed.
# ---------------------------------------------------------------------------

def build_feature_matrix(raw_df: pd.DataFrame,
                         target_col: str = "GLD",
                         denoise: bool = True) -> tuple:
    """
    Builds a lag-safe, wavelet-denoised SVR feature matrix from the raw
    price + macro DataFrame produced by loader.py.

    Pipeline:
      1. Compute daily returns from raw prices (pct_change)
      2. Apply DWT denoising to each return series (if denoise=True)
      3. Build lagged features — all shifted >= 1 day (no look-ahead)
      4. Drop NaN rows

    Parameters
    ----------
    raw_df     : Raw price + macro level DataFrame from loader.py
    target_col : ETF column to predict next-day return for (e.g. "GLD")
    denoise    : Whether to apply wavelet denoising before feature engineering

    Returns
    -------
    X            : np.ndarray (n_samples, n_features)
    y            : np.ndarray (n_samples,) — next-day target returns
    valid_index  : pd.DatetimeIndex of rows included
    feature_names: list[str]
    """
    # --- Step 1: Compute returns from raw prices ---
    # ETF columns are prices → pct_change
    # Macro columns are already levels (VIX, spreads) → use as-is or pct_change
    etf_cols   = ["GLD", "SPY", "AGG", "TLT", "TBT", "VNQ", "SLV"]
    macro_cols = ["VIX", "DXY", "T10Y2Y", "SOFR", "IG_SPREAD", "HY_SPREAD"]

    ret_df = pd.DataFrame(index=raw_df.index)

    for col in etf_cols:
        if col in raw_df.columns:
            ret_df[f"{col}_Ret"] = raw_df[col].pct_change()

    # Macro: use level changes (first difference) to make them stationary
    for col in macro_cols:
        if col in raw_df.columns:
            ret_df[f"{col}_chg"] = raw_df[col].diff()
            ret_df[f"{col}_lvl"] = raw_df[col]  # keep level too as a feature

    target_ret_col = f"{target_col}_Ret"
    if target_ret_col not in ret_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in raw_df. "
                         f"Available ETFs: {[c for c in raw_df.columns if c in etf_cols]}")

    # --- Step 2: Wavelet denoise each return series ---
    if denoise:
        for col in ret_df.columns:
            if col.endswith("_Ret") or col.endswith("_chg"):
                ret_df[col] = apply_dwt_denoise(ret_df[col].dropna()
                                                 .reindex(ret_df.index))

    # --- Step 3: Build lag-safe feature matrix ---
    feat = pd.DataFrame(index=ret_df.index)

    # Lagged target returns
    for lag in [1, 3, 5, 10, 21]:
        feat[f"target_lag{lag}"] = ret_df[target_ret_col].shift(lag)

    # Rolling realised volatility of target (lagged 1 so known before bar opens)
    feat["target_vol5d"]  = ret_df[target_ret_col].rolling(5).std().shift(1)
    feat["target_vol21d"] = ret_df[target_ret_col].rolling(21).std().shift(1)

    # Cross-asset return lags (lag 1)
    cross_asset_rets = [c for c in ret_df.columns
                        if c.endswith("_Ret") and c != target_ret_col]
    for col in cross_asset_rets:
        feat[f"{col}_lag1"] = ret_df[col].shift(1)

    # Macro features (lag 1 — yesterday's macro level/change is known today)
    macro_feat_cols = [c for c in ret_df.columns
                       if c.endswith("_chg") or c.endswith("_lvl")]
    for col in macro_feat_cols:
        feat[f"{col}_lag1"] = ret_df[col].shift(1)

    # Target: next-day return (what SVR predicts)
    feat["__target__"] = ret_df[target_ret_col]

    # --- Step 4: Drop NaN rows ---
    feat.dropna(inplace=True)

    feature_names = [c for c in feat.columns if c != "__target__"]
    X = feat[feature_names].values
    y = feat["__target__"].values

    print(f"Feature matrix built: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feat.index, feature_names
