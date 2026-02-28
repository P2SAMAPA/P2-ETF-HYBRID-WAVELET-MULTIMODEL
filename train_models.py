import os
import numpy as np
import pandas as pd
import shutil

# --- FORCE CPU USAGE (Critical for GitHub Actions Free Tier) ---
try:
    import tensorflow as tf
    # Disable GPU to prevent memory allocation errors on limited environments
    tf.config.set_visible_devices([], 'GPU')
    print("✅ TensorFlow configured to CPU only.")
except ImportError:
    print("ℹ️ TensorFlow not found.")

from data.loader import load_raw_data
from data.processor import build_feature_matrix
from engine import DeepHybridEngine

def train_and_save_all():
    # 1. Clear old models and ensure directory exists fresh
    if os.path.exists("models"):
        shutil.rmtree("models")
    os.makedirs("models", exist_ok=True)
    
    # 2. Setup environment and Load Data
    print("Downloading data from Hugging Face...")
    df, _ = load_raw_data(force_sync=False)
    
    if df is None or df.empty:
        print("❌ Error: Loaded DataFrame is empty. Check dataset connection.")
        return

    # 3. MULTI-ETF CONCATENATION LOGIC
    # These are the 7 ETFs your model must understand to avoid 'TBT' errors in app.py
    target_assets = ["TLT", "LQD", "HYG", "VCIT", "VNQ", "GLD", "SLV"]
    
    all_X_3d = []
    all_y_3d = []
    all_X_macro = [] # For Option K
    
    lookback = 20
    print(f"🔄 Processing features for {len(target_assets)} ETFs...")

    for ticker in target_assets:
        try:
            # Build features for specific ticker
            X, y, idx, _ = build_feature_matrix(df, target_col=ticker)
            
            # Create 3D Windows (Tensors) for Deep Learning
            n_samples = len(X) - lookback + 1
            n_features = X.shape
            
            X_ticker_3d = np.zeros((n_samples, lookback, n_features))
            y_ticker_3d = np.zeros(n_samples)
            
            for i in range(lookback, len(X) + 1):
                X_ticker_3d[i - lookback] = X[i - lookback:i]
                y_ticker_3d[i - lookback] = y[i - 1]
            
            all_X_3d.append(X_ticker_3d)
            all_y_3d.append(y_ticker_3d)
            
            # Extract Macro features for the same indices (Option K support)
            # We use VIX, DXY, T10Y2Y, IG_SPREAD, HY_SPREAD as global context
            macro_cols = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]
            if all(c in df.columns for c in macro_cols):
                m_df = df[macro_cols].iloc[lookback-1:].copy()
                # Match shape by adding dummy columns used by the engine
                m_df["Mom"], m_df["Vol"], m_df["SPY"] = 0.0, 0.0, 0.0
                all_X_macro.append(m_df.values)
                
            print(f"  ✅ {ticker}: Prepared {n_samples} samples.")
        except Exception as e:
            print(f"  ❌ Skipping {ticker} due to error: {e}")

    if not all_X_3d:
        print("❌ Error: No training data generated.")
        return

    # Concatenate all assets into one global training set
    X_train_final = np.concatenate(all_X_3d, axis=0)
    y_train_final = np.concatenate(all_y_3d, axis=0)
    X_macro_final = np.concatenate(all_X_macro, axis=0) if all_X_macro else None

    # 4. SHUFFLE DATA
    # Shuffling is critical so the model doesn't learn assets in a specific order
    indices = np.arange(len(X_train_final))
    np.random.shuffle(indices)
    
    X_train_final = X_train_final[indices]
    y_train_final = y_train_final[indices]
    if X_macro_final is not None:
        X_macro_final = X_macro_final[indices]

    print(f"📊 Global Dataset Ready: {X_train_final.shape} total samples.")

    # 5. TRAIN AND SAVE MODELS
    # --- Option I ---
    print("\n🚀 Training Option I (CNN)...")
    model_i = DeepHybridEngine(mode="Option I")
    # Training on global set; logic for I/J ignores macro_in
    if model_i.train(X_train_final, y_train_final):
        model_i.save("models/opt_i_cnn.h5")
        print("  💾 Saved: models/opt_i_cnn.h5")

    # --- Option J ---
    print("\n🚀 Training Option J (Attention-CNN-LSTM)...")
    model_j = DeepHybridEngine(mode="Option J")
    if model_j.train(X_train_final, y_train_final):
        model_j.save("models/opt_j_cnn_lstm.h5")
        print("  💾 Saved: models/opt_j_cnn_lstm.h5")

    # --- Option K ---
    print("\n🚀 Training Option K (Parallel Dual-Stream)...")
    model_k = DeepHybridEngine(mode="Option K")
    # Option K requires both Price windows and Macro features
    if model_k.train(X_train_final, y_train_final, X_macro=X_macro_final):
        model_k.save("models/opt_k_hybrid.h5")
        print("  💾 Saved: models/opt_k_hybrid.h5")

    print("\n✅ All Cloud Models (I, J, K) have been updated for the 7-ETF Universe.")

if __name__ == "__main__":
    train_and_save_all()
