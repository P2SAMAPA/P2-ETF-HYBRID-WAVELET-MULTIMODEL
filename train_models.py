import os
import numpy as np
import pandas as pd
from data.loader import load_raw_data
from data.processor import build_feature_matrix
from models.engine import DeepHybridEngine

def automate_training():
    # 1. Load Data
    df = load_raw_data()
    lookback = 20
    
    # Target column - update to whatever your dashboard is focusing on (e.g., 'SLV' or 'GLD')
    target = "SLV"
    X, y, _, _ = build_feature_matrix(df, target_col=target)
    
    # 2. Reshape for CNN-LSTM (3D Tensor)
    X_3d = np.array([X[i-lookback:i] for i in range(lookback, len(X))])
    y_adj = y[lookback:]
    X_macro = df[['HY_SPREAD', 'VIX', 'DXY']].iloc[lookback:].values

    # Ensure /models directory exists
    os.makedirs("models", exist_ok=True)

    # 3. Train and Save Option K (Dual-Stream)
    print("Training Option K...")
    engine_k = DeepHybridEngine(mode="Option K", lookback=lookback)
    if engine_k.train(X_3d, y_adj, X_macro=X_macro):
        engine_k.save("models/opt_k_dual.h5")

    # 4. Train Option I (Wavelet-CNN-LSTM)
    print("Training Option I...")
    engine_i = DeepHybridEngine(mode="Option I", lookback=lookback)
    if engine_i.train(X_3d, y_adj):
        engine_i.save("models/opt_i_cnn.h5")

    print("✅ Training Complete. Files generated in /models.")
    # --- DIRECT API INJECTION ---
    from huggingface_hub import HfApi
    import os

    api = HfApi()
    token = os.getenv("HF_TOKEN")

    if token:
    print("🚀 Teleporting models to Hugging Face...")
    api.upload_folder(
        folder_path="models",
        repo_id="P2SAMAPA/P2-ETF-HYBRID-WAVELET-PPO",
        repo_type="space",
        path_in_repo="models",  # This puts files inside the /models folder
        token=token
    )
    print("✅ Injection complete!")
    else:
    print("❌ HF_TOKEN missing. Skipping upload.")

if __name__ == "__main__":
    automate_training()
