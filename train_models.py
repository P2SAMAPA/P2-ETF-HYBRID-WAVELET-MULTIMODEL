import os
import numpy as np
import pandas as pd
import shutil

# --- FORCE CPU USAGE (Critical for GitHub Actions Free Tier) ---
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    print("✅ TensorFlow configured to CPU only.")
except ImportError:
    print("ℹ️ TensorFlow not found (using other backend).")

try:
    import torch
    if torch.cuda.is_available():
        print("⚠️ Warning: CUDA detected but should not be available on GitHub Actions.")
    device = torch.device("cpu")
    print("✅ PyTorch configured to CPU.")
except ImportError:
    pass
# ---------------------------------------------------------------

from data.loader import load_raw_data
from data.processor import build_feature_matrix
from engine import DeepHybridEngine, MomentumEngine, A2CEngine

def train_and_save_all():
    # 1. Clear old models and ensure directory exists fresh
    if os.path.exists("models"):
        shutil.rmtree("models")
    os.makedirs("models", exist_ok=True)
    
    # 2. Setup environment for the loader
    print("Preparing data environment...")

    # 3. Load Data
    print("Downloading data from Hugging Face...")
    df, _ = load_raw_data(force_sync=False)
    
    if df is None or df.empty:
        print("❌ Error: Loaded DataFrame is empty. Check HF_TOKEN and FRED_API_KEY.")
        return

    # 4. Build features using your processor.py
    print(f"Processing features for {len(df)} rows...")
    X, y, _, feature_names = build_feature_matrix(df)

    # === CONFIRM NEW FEATURE COUNT (9 ETFs + macros - TBT replaced) ===
    n_features = X.shape[1]
    print(f"✅ Feature matrix built with {n_features} features (GLD, SPY, AGG, TLT, VCIT, LQD, HYG, VNQ, SLV + macros)")
    print(f"   Feature names: {feature_names}")

    # DL models require 3D input (Samples, Lookback, Features)
    lookback = 20
    
    # --- MEMORY OPTIMIZATION ---
    n_samples = len(X) - lookback + 1
    X_3d = np.zeros((n_samples, lookback, n_features))
    y_3d = np.zeros(n_samples)
    
    for i in range(lookback, len(X) + 1):
        X_3d[i - lookback] = X[i - lookback:i]
        y_3d[i - lookback] = y[i - 1]
    # ---------------------------

    # --- Train Option I ---
    print("Training Option I (CNN)...")
    model_i = DeepHybridEngine(mode="Option I")
    if model_i.train(X_3d, y_3d):
        model_i.save("models/opt_i_cnn.h5")

    # --- Train Option J ---
    print("Training Option J (CNN-LSTM)...")
    model_j = DeepHybridEngine(mode="Option J")
    if model_j.train(X_3d, y_3d):
        model_j.save("models/opt_j_cnn_lstm.h5")

    # --- Train Option K ---
    print("Training Option K (Hybrid)...")
    model_k = DeepHybridEngine(mode="Option K")
    if model_k.train(X_3d, y_3d):
        model_k.save("models/opt_k_hybrid.h5")
    
    # --- Train Option C ---
    print("Training Option C (A2C)...")
    model_c = A2CEngine()
    if model_c.train(X, y):
        model_c.save("models/a2c_weights.pkl")
    
    # --- Train Option A ---
    print("Updating Option A (SVR)...")
    model_a = MomentumEngine()
    if model_a.train(X, y):
        model_a.save("models/svr_momentum_poly.pkl")

    print("✅ All models trained and saved to /models")

if __name__ == "__main__":
    train_models_and_save_all()
