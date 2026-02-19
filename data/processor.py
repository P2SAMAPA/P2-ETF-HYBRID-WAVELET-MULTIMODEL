import pywt
import numpy as np
import pandas as pd
import warnings


# ---------------------------------------------------------------------------
# CORE WAVELET DENOISER
# ---------------------------------------------------------------------------

def _modwt_decompose(data: np.ndarray, wavelet: str, level: int):
    """
    True MODWT (Maximal Overlap DWT) via pywt.swt (Stationary Wavelet Transform).
    Unlike pywt.wavedec (standard DWT which downsamples), swt is shift-invariant:
    the reconstructed signal phase does not depend on where the series starts.
    This is critical for financial time series where timing matters.

    Returns (coeffs, n_orig) where coeffs is a list of (cA, cD) pairs.
    """
    n = len(data)
    # swt requires length to be a multiple of 2^level
    required = int(2 ** level)
    pad = required - (n % required)
    pad = 0 if pad == required else pad
    padded = np.pad(data, (0, pad), mode='reflect') if pad > 0 else data.copy()
    coeffs = pywt.swt(padded, wavelet, level=level, trim_approx=False)
    return coeffs, n


def _universal_threshold(finest_detail: np.ndarray, n: int) -> float:
    """
    VisuShrink universal threshold: sigma * sqrt(2 * log(n))
    Sigma estimated via median absolute deviation — robust to outliers.
    """
    sigma = np.median(np.abs(finest_detail)) / 0.6745
    return float(sigma * np.sqrt(2 * np.log(max(n, 2))))


def _soft_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    """Soft thresholding — preserves partial signal, avoids over-smoothing."""
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)


def apply_modwt_denoise(series: pd.Series, level: int = 3,
                        wavelet: str = 'sym4') -> pd.Series:
    """
    Applies true MODWT denoising to a return series.

    Fixes vs original code:
      - Uses pywt.swt (shift-invariant) instead of pywt.wavedec (DWT, not MODWT)
      - Soft thresholding on detail coefficients instead of zeroing them entirely
      - float64 instead of float32 (daily returns are small — precision matters)
      - Input validation for short or all-NaN series

    Parameters
    ----------
    series  : pd.Series of daily returns. Index must be DatetimeIndex.
    level   : Decomposition depth. 3 is appropriate for daily return series.
    wavelet : 'sym4' is well-suited for smooth financial signals.

    Returns
    -------
    pd.Series of denoised returns, same length and index as input.
    """
    # --- Input validation ---
    n_valid = series.dropna().shape[0]
    min_required = int(2 ** level) * 2

    if n_valid == 0:
        return series.copy()

    if n_valid < min_required:
        warnings.warn(
            f"apply_modwt_denoise: only {n_valid} non-NaN rows — need at least "
            f"{min_required} for level={level}. Returning original series."
        )
        return series.copy()

    # Work in float64 — daily returns are small numbers, float32 risks rounding
    data = series.fillna(0.0).values.astype(np.float64)
    n_orig = len(data)

    # --- MODWT decomposition ---
    coeffs, _ = _modwt_decompose(data, wavelet, level)
    # coeffs: list of (cA, cD) tuples, index 0 = coarsest, -1 = finest

    # --- Compute threshold from finest detail layer ---
    finest_detail = coeffs[-1][1][:n_orig]
    threshold = _universal_threshold(finest_detail, n_orig)

    # --- Soft threshold each detail layer independently ---
    thresholded = [(cA, _soft_threshold(cD, threshold)) for cA, cD in coeffs]

    # --- Reconstruct via inverse SWT ---
    reconstructed = pywt.iswt(thresholded, wavelet)
    denoised = reconstructed[:n_orig]

    return pd.Series(denoised, index=series.index, name=series.name)


# ---------------------------------------------------------------------------
# FEATURE BUILDER — called by engine.py and app.py
# ---------------------------------------------------------------------------

def build_feature_matrix(raw_df: pd.DataFrame,
                         target_col: str = "GLD",
                         denoise_level: int = 3) -> tuple:
    """
    Transforms raw loader.py output (ETF prices + macro levels) into a
    lag-safe, wavelet-denoised SVR feature matrix.

    Pipeline:
      1. Compute daily returns from ETF prices (.pct_change())
      2. Apply MODWT denoising to each ETF return series
      3. Build lagged feature matrix — all lags >= 1 day (zero look-ahead bias)
      4. Add rolling realised volatility features (also lagged 1 day)
      5. Add macro level features (lagged 1 day)
      6. Target = next-day return of target_col (no lag — this is what SVR predicts)

    Parameters
    ----------
    raw_df       : DataFrame from loader.py — ETF raw prices + macro levels.
    target_col   : ETF to predict (default 'GLD').
    denoise_level: Wavelet decomposition depth (default 3).

    Returns
    -------
    X             : np.ndarray (n_samples, n_features)
    y             : np.ndarray (n_samples,)
    valid_index   : pd.DatetimeIndex aligned with X and y
    feature_names : list[str]
    """
    etf_cols   = ["GLD", "SLV", "SPY", "AGG", "TLT", "TBT", "VNQ"]
    macro_cols = ["VIX", "DXY", "T10Y2Y", "SOFR", "IG_OAS", "HY_OAS"]

    ret_cols      = [c for c in etf_cols   if c in raw_df.columns]
    mac_available = [c for c in macro_cols if c in raw_df.columns]

    if target_col not in ret_cols:
        raise ValueError(f"target_col '{target_col}' not found in raw_df columns.")

    # --- Step 1: ETF returns ---
    returns = raw_df[ret_cols].pct_change()

    # --- Step 2: Denoise each ETF return series ---
    denoised = pd.DataFrame(index=returns.index)
    for col in ret_cols:
        denoised[col] = apply_modwt_denoise(returns[col], level=denoise_level)

    # --- Step 3 & 4: Lagged features (all shifted >= 1 day — no look-ahead) ---
    feat = pd.DataFrame(index=raw_df.index)

    for lag in [1, 3, 5, 10, 21]:
        for col in ret_cols:
            feat[f"{col}_ret_lag{lag}"] = denoised[col].shift(lag)

    for window in [5, 21]:
        feat[f"{target_col}_vol_{window}d"] = (
            returns[target_col].rolling(window).std().shift(1)
        )

    # --- Step 5: Macro levels lagged 1 day ---
    for col in mac_available:
        feat[f"{col}_lag1"] = raw_df[col].shift(1)

    # --- Step 6: Target (next-day return — no lag) ---
    feat["__target__"] = returns[target_col]

    feat.dropna(inplace=True)

    feature_names = [c for c in feat.columns if c != "__target__"]
    X = feat[feature_names].values.astype(np.float64)
    y = feat["__target__"].values.astype(np.float64)

    return X, y, feat.index, feature_names
