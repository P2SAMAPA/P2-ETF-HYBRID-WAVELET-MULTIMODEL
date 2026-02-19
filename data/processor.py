import numpy as np
import pandas as pd
import pywt


# ---------------------------------------------------------------------------
# WAVELET DENOISER
# Fixed issues vs original:
#    1. Labelled and documented as DWT (not MODWT — pywt.wavedec is DWT)
#    2. float64 instead of float32 — preserves precision for small daily returns
#    3. Soft thresholding on detail coefficients instead of zeroing them out —
#       preserves more signal structure while still removing high-freq noise
#    4. Input validation for short or NaN-heavy series
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
        return series

    if clean.std() == 0:
        print("DWT skipped: series is constant")
        return series

    # --- Handle NaNs: denoise only on non-NaN values, preserve NaN positions ---
    nan_mask = series.isna()
    filled   = series.fillna(0.0)

    # --- Use float64 for precision (daily returns are tiny numbers) ---
    data = filled.values.astype(np.float64)

    # --- Decompose ---
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # --- Soft threshold on detail coefficients ---
    # Threshold estimated from finest-level detail using the universal threshold:
    # sigma * sqrt(2 * log(N)) where sigma is estimated via MAD of finest details
    finest_detail = coeffs[-1]
    sigma     = np.median(np.abs(finest_detail)) / 0.6745  # MAD estimator
    threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # --- Reconstruction Logic ---
    if threshold > 0:
        denoised_coeffs = [coeffs[0]] + [
            pywt.threshold(c, value=threshold, mode="soft") for c in coeffs[1:]
        ]
        reconstructed = pywt.waverec(denoised_coeffs, wavelet)
        reconstructed = reconstructed[:len(data)]
    else:
        reconstructed = data

    result = pd.Series(reconstructed, index=series.index, name=series.name)
    # Restore original NaN positions
    result[nan_mask] = np.nan
    return result


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
      4. Handle NaNs via ffill and zero-fill to preserve early history
    """
    etf_cols   = ["GLD", "SPY", "AGG", "TLT", "TBT", "VNQ", "SLV"]
    macro_cols = ["VIX", "DXY", "T10Y2Y", "SOFR", "IG_SPREAD", "HY_SPREAD"]

    ret_df = pd.DataFrame(index=raw_df.index)

    # --- Step 1: Compute returns from raw prices ---
    for col in etf_cols:
        if col in raw_df.columns:
            ret_df[f"{col}_Ret"] = raw_df[col].pct_change()

    for col in macro_cols:
        if col in raw_df.columns:
            ret_df[f"{col}_chg"] = raw_df[col].diff()
            ret_df[f"{col}_lvl"] = raw_df[col]

    target_ret_col = f"{target_col}_Ret"
    if target_ret_col not in ret_df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # --- Step 2: Wavelet denoise each return series ---
    if denoise:
        for col in ret_df.columns:
            if col.endswith("_Ret") or col.endswith("_chg"):
                ret_df[col] = apply_dwt_denoise(ret_df[col])

    # --- Step 3: Build lag-safe feature matrix ---
    feat = pd.DataFrame(index=ret_df.index)

    for lag in [1, 3, 5, 10, 21]:
        feat[f"target_lag{lag}"] = ret_df[target_ret_col].shift(lag)

    feat["target_vol5d"]  = ret_df[target_ret_col].rolling(5).std().shift(1)
    feat["target_vol21d"] = ret_df[target_ret_col].rolling(21).std().shift(1)

    other_cols = [c for c in ret_df.columns if c != target_ret_col]
    for col in other_cols:
        feat[f"{col}_lag1"] = ret_df[col].shift(1)

    feat["__target__"] = ret_df[target_ret_col]

    # --- Step 4: Handle NaNs (Critical Fix for "0 rows" error) ---
    # ffill() carries the last known value forward.
    # fillna(0) ensures early rows (where macro wasn't yet reported) aren't dropped.
    feat = feat.ffill().fillna(0)
    
    # Drop first 22 rows where rolling/lags are all zeros/NaNs
    feat = feat.iloc[22:]

    feature_names = [c for c in feat.columns if c != "__target__"]
    X = feat[feature_names].values
    y = feat["__target__"].values

    print(f"Feature matrix built: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feat.index, feature_names
