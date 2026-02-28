import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler

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

def build_feature_matrix(raw_df: pd.DataFrame, target_col: str = "GLD", denoise: bool = True) -> tuple:
    # ── ETF LIST (TBT removed, VCIT/LQD/HYG added) ───────────────────────────────
    etf_cols   = ["GLD", "SPY", "AGG", "TLT", "VCIT", "LQD", "HYG", "VNQ", "SLV"]
    macro_cols = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]

    # Safety check: warn if any expected column is completely missing
    missing_etfs = [col for col in etf_cols if col not in raw_df.columns]
    if missing_etfs:
        print(f"Warning: Missing ETF columns in raw data: {missing_etfs}")

    ret_df = pd.DataFrame(index=raw_df.index)
    for col in etf_cols:
        if col in raw_df.columns:
            ret_df[f"{col}_Ret"] = raw_df[col].pct_change()
        else:
            ret_df[f"{col}_Ret"] = np.nan  # explicit NaN column

    for col in macro_cols:
        if col in raw_df.columns:
            ret_df[f"{col}_lvl"] = raw_df[col]

    target_ret_col = f"{target_col}_Ret"

    if denoise:
        for col in ret_df.columns:
            if "_Ret" in col:
                ret_df[col] = apply_dwt_denoise(ret_df[col])

    feat = pd.DataFrame(index=ret_df.index)
    for lag in [1, 3, 5, 10, 21]:
        feat[f"target_lag{lag}"] = ret_df[target_ret_col].shift(lag)

    for col in macro_cols:
        if f"{col}_lvl" in ret_df.columns:
            feat[f"{col}_lag1"] = ret_df[f"{col}_lvl"].shift(1)

    feat["__target__"] = raw_df[target_col].pct_change()
    feat = feat.dropna().iloc[22:]

    feature_names = [c for c in feat.columns if c != "__target__"]
    X_raw = feat[feature_names].values

    # Standardize features for model stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, feat["__target__"].values, feat.index, feature_names
