import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]

def apply_dwt_denoise(series: pd.Series, wavelet: str = "sym4", level: int = 3) -> pd.Series:
    clean = series.dropna()
    if len(clean) < (2 ** level):
        return series
    if clean.std() == 0:
        return series

    nan_mask = series.isna()
    data = series.fillna(0.0).values.astype(np.float64)
    coeffs = pywt.wavedec(data, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))

    if threshold > 0:
        denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, value=threshold, mode="soft") for c in coeffs[1:]]
        reconstructed = pywt.waverec(denoised_coeffs, wavelet)[:len(data)]
    else:
        reconstructed = data

    result = pd.Series(reconstructed, index=series.index, name=series.name)
    result[nan_mask] = np.nan
    return result

def build_feature_matrix(raw_df: pd.DataFrame, target_col: str, feature_symbols: list, denoise: bool = True) -> tuple:
    """
    Build feature matrix for a given target asset, using returns of the provided feature_symbols
    (the set of assets in the same category, including possibly the target itself).
    """
    if target_col not in raw_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in raw data")

    # Build return DataFrame for all feature symbols
    ret_df = pd.DataFrame(index=raw_df.index)
    for sym in feature_symbols:
        if sym in raw_df.columns:
            ret_df[f"{sym}_Ret"] = raw_df[sym].pct_change()
        else:
            ret_df[f"{sym}_Ret"] = np.nan

    # Add macro levels
    for col in MACRO_COLS:
        if col in raw_df.columns:
            ret_df[f"{col}_lvl"] = raw_df[col]

    if denoise:
        for col in ret_df.columns:
            if "_Ret" in col:
                ret_df[col] = apply_dwt_denoise(ret_df[col])

    # Build lagged features
    feat = pd.DataFrame(index=ret_df.index)
    # Lags of target returns
    for lag in [1, 3, 5, 10, 21]:
        feat[f"target_lag{lag}"] = ret_df[f"{target_col}_Ret"].shift(lag)

    # Lagged macro levels
    for col in MACRO_COLS:
        if f"{col}_lvl" in ret_df.columns:
            feat[f"{col}_lag1"] = ret_df[f"{col}_lvl"].shift(1)

    # Add returns of all feature symbols except target as features
    for sym in feature_symbols:
        if sym == target_col:
            continue
        feat[f"{sym}_ret"] = ret_df[f"{sym}_Ret"]

    feat["__target__"] = raw_df[target_col].pct_change()
    feat = feat.dropna().iloc[22:]

    feature_names = [c for c in feat.columns if c != "__target__"]
    X_raw = feat[feature_names].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, feat["__target__"].values, feat.index, feature_names
