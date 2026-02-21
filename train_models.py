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
    
    # Target column
    target = "SLV"
    X, y, _, _ = build_feature_matrix(df, target_col=target)
    
    # 2. Reshape for CNN-LSTM (3D Tensor)
    X_3d = np.array([X[i-lookback:i] for i in range(lookback, len(X))])
    y_adj = y[lookback:]
    X_macro = df[['HY_SPREAD', 'VIX', 'DXY']].iloc[lookback:].values

    # Ensure /models directory exists locally
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
    
    api = HfApi()
    token = os.getenv("HF_TOKEN")
    repo_id = "P2SAMAPA/P2-ETF-HYBRID-WAVELET-PPO"
    
    if token:
        print("🚀 Teleporting models to Hugging Face models/ folder...")
        try:
            # Force upload to the SPECIFIC subfolder
            for model_file in ["opt_k_dual.h5", "opt_i_cnn.h5"]:
                local_path = f"models/{model_file}"
                if os.path.exists(local_path):
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=f"models/{model_file}", # This is the "GPS" fix
                        repo_id=repo_id,
                        repo_type="space",
                        token=token
                    )
                    print(f"✅ {model_file} injected into models/ successfully!")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
    else:
        print("⚠️ HF_TOKEN missing. Skipping upload.")

if __name__ == "__main__":
    automate_training()
