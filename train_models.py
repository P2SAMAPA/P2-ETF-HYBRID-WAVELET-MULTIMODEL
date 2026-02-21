import os
import numpy as np
import pandas as pd
from data.processor import load_raw_data, build_feature_matrix
from engine import DeepHybridEngine, MomentumEngine

def train_and_save_all():
    # 1. Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # 2. Load Data
    df = load_raw_data()
    X, y, X_macro, _ = build_feature_matrix(df)
    
    # DL models require 3D input (Samples, Lookback, Features)
    lookback = 20
    X_3d = np.array([X[i-lookback:i] for i in range(lookback, len(X)+1)])
    y_3d = y[lookback-1:]
    
    # --- Train Option I (CNN) ---
    print("Training Option I...")
    model_i = DeepHybridEngine(mode="Option I")
    if model_i.train(X_3d, y_3d):
        model_i.save("models/opt_i_cnn.h5")

    # --- Train Option J (CNN-LSTM) ---
    print("Training Option J...")
    model_j = DeepHybridEngine(mode="Option J")
    if model_j.train(X_3d, y_3d):
        model_j.save("models/opt_j_cnn_lstm.h5")

    # --- Train Option K (Hybrid) ---
    print("Training Option K...")
    model_k = DeepHybridEngine(mode="Option K")
    # Align macro data with windowed price data
    X_macro_k = X_macro[lookback-1:] 
    if model_k.train(X_3d, y_3d, X_macro=X_macro_k):
        model_k.save("models/opt_k_hybrid.h5")

    # --- Train Option A (SVR) ---
    print("Updating Option A...")
    model_a = MomentumEngine()
    if model_a.train(X, y):
        model_a.save("models/svr_momentum_poly.pkl")

if __name__ == "__main__":
    train_and_save_all()
