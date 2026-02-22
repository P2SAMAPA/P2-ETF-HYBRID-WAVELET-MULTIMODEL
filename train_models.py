import os
import numpy as np
import pandas as pd
# Import from your specific file structure
from data.loader import load_raw_data 
from data.processor import build_feature_matrix
from engine import DeepHybridEngine, MomentumEngine, A2CEngine

def train_and_save_all():
    # 1. Clear old models and ensure directory exists fresh
    import shutil
    if os.path.exists("models"):
        shutil.rmtree("models")
    os.makedirs("models", exist_ok=True)
    
    # 2. Setup environment for the loader
    # We map GitHub Secrets to the names loader.py expects in its fallback
    print("Preparing data environment...")
    
    # 3. Load Data
    # We set force_sync=False because the trainer just needs to read the master data
    print("Downloading data from Hugging Face...")
    df, _ = load_raw_data(force_sync=False)
    
    # Note the ', _' which catches the status message and prevents the error
    df, _ = load_raw_data() 
    if df.empty:
    print("❌ Error: Loaded DataFrame is empty. Check HF_TOKEN and FRED_API_KEY.")
    return

    # 4. Build features using your processor.py
    print(f"Processing features for {len(df)} rows...")
    X, y, _, _ = build_feature_matrix(df)
    
    # DL models require 3D input (Samples, Lookback, Features)
    lookback = 20
    X_3d = np.array([X[i-lookback:i] for i in range(lookback, len(X)+1)])
    y_3d = y[lookback-1:]
    
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
    train_and_save_all()
