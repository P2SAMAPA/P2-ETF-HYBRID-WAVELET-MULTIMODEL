import pywt
import numpy as np
import pandas as pd

def apply_modwt_denoise(series, level=3):
    """
    Applies MODWT denoising. 
    Uses 'sym4' wavelet and recovers the 'Trend' component.
    """
    data = series.values.astype(np.float32)
    # MODWT implementation using PyWavelets multilevel decomposition
    coeffs = pywt.wavedec(data, 'sym4', level=level)
    
    # Reconstruct only using the approximation coefficients (the Trend)
    # This removes high-frequency noise while keeping the signal time-aligned
    reconstructed = pywt.waverec([coeffs[0]] + [None] * (len(coeffs) - 1), 'sym4')
    
    # Ensure lengths match due to padding
    return pd.Series(reconstructed[:len(data)], index=series.index)
