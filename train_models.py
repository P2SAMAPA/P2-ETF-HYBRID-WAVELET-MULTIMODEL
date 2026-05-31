import os
import numpy as np
import pandas as pd
import shutil
import yaml

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    print("✅ TensorFlow configured to CPU only.")
except ImportError:
    print("ℹ️ TensorFlow not found.")

# ---------------------------------------------------------------------------
# Direct HF data load — bypasses st.secrets / Streamlit entirely
# load_raw_data uses st.secrets which fails in GitHub Actions context,
# causing unauthenticated HF requests that hit rate limits and return
# stale cached parquet (~743 rows instead of ~4800).
# Fix: load the parquet directly via hf_hub_download with env token.
# ---------------------------------------------------------------------------
from huggingface_hub import hf_hub_download
from data.processor import build_feature_matrix, get_canonical_feature_names
from engine import DeepHybridEngine


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_data_direct() -> pd.DataFrame:
    """
    Load master_data.parquet directly from HuggingFace using HF_TOKEN
    environment variable. Bypasses Streamlit cache and st.secrets entirely.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  HF_TOKEN not set — downloading without auth (rate limits apply)")

    print(f"  Auth: {'✅ token present' if hf_token else '⚠️  no token'}")

    path = hf_hub_download(
        repo_id="P2SAMAPA/fi-etf-macro-signal-master-data",
        filename="master_data.parquet",
        repo_type="dataset",
        token=hf_token,
        force_download=True,   # bypass local cache — always get latest file
    )
    df = pd.read_parquet(path)

    # Normalise index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    print(f"  Loaded: {len(df)} rows × {len(df.columns)} columns "
          f"({df.index.min().date()} → {df.index.max().date()})")

    # Add _Ret columns (needed by some downstream code)
    config   = load_config()
    symbols  = config['seeding']['symbols']
    for t in symbols:
        if t in df.columns:
            df[f"{t}_Ret"] = df[t].ffill().pct_change(fill_method=None)

    return df


def train_category(category_name, asset_list, raw_df, lookback=20):
    print(f"\n🔄 Processing category: {category_name} with {len(asset_list)} assets...")

    canonical_names = get_canonical_feature_names(
        raw_df=raw_df,
        feature_symbols=asset_list,
        include_options=True,
        include_volume=True,
        market_proxy="SPY",
    )
    n_features = len(canonical_names)
    print(f"  Canonical feature width: {n_features} columns")

    all_X_3d    = []
    all_y_3d    = []
    all_X_macro = []

    for ticker in asset_list:
        try:
            X, y, idx, feat_names = build_feature_matrix(
                raw_df,
                target_col=ticker,
                feature_symbols=asset_list,
                canonical_names=canonical_names,
            )

            n_samples = len(X) - lookback + 1
            if n_samples <= 0:
                print(f"  ⚠️  {ticker}: Not enough data for window {lookback}. Skipping.")
                continue

            X_ticker_3d = np.zeros((n_samples, lookback, n_features))
            y_ticker_3d = np.zeros(n_samples)
            for i in range(lookback, len(X) + 1):
                X_ticker_3d[i - lookback] = X[i - lookback:i]
                y_ticker_3d[i - lookback] = y[i - 1]

            all_X_3d.append(X_ticker_3d)
            all_y_3d.append(y_ticker_3d)

            macro_cols = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]
            if all(c in raw_df.columns for c in macro_cols):
                m_df = raw_df[macro_cols].iloc[lookback - 1:].copy()
                for dummy in ["Mom", "Vol", "SPY"]:
                    m_df[dummy] = 0.0
                if len(m_df) >= n_samples:
                    m_df = m_df.iloc[:n_samples]
                else:
                    pad = pd.DataFrame(
                        np.zeros((n_samples - len(m_df), len(macro_cols) + 3)),
                        columns=m_df.columns
                    )
                    m_df = pd.concat([m_df, pad], axis=0)
                all_X_macro.append(m_df.values)

            print(f"  ✅ {ticker}: Prepared {n_samples} samples ({n_features} features).")

        except Exception as e:
            print(f"  ❌ Skipping {ticker} due to error: {e}")
            import traceback; traceback.print_exc()

    if not all_X_3d:
        print(f"❌ No training data for category {category_name}. Skipping.")
        return

    X_train = np.concatenate(all_X_3d, axis=0)
    y_train = np.concatenate(all_y_3d, axis=0)
    X_macro = np.concatenate(all_X_macro, axis=0) if all_X_macro else None

    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    if X_macro is not None:
        X_macro = X_macro[indices]

    print(f"📊 {category_name} global dataset: {X_train.shape} samples.")

    print(f"\n🚀 Training Option I (CNN) for {category_name}...")
    model_i = DeepHybridEngine(mode="Option I")
    if model_i.train(X_train, y_train):
        model_i.save(f"models/{category_name}_opt_i_cnn.h5")
        print(f"  💾 Saved: models/{category_name}_opt_i_cnn.h5")

    print(f"\n🚀 Training Option J (Attention-CNN-LSTM) for {category_name}...")
    model_j = DeepHybridEngine(mode="Option J")
    if model_j.train(X_train, y_train):
        model_j.save(f"models/{category_name}_opt_j_cnn_lstm.h5")
        print(f"  💾 Saved: models/{category_name}_opt_j_cnn_lstm.h5")

    print(f"\n🚀 Training Option K (Parallel Dual-Stream) for {category_name}...")
    model_k = DeepHybridEngine(mode="Option K")
    if model_k.train(X_train, y_train, X_macro=X_macro):
        model_k.save(f"models/{category_name}_opt_k_hybrid.h5")
        print(f"  💾 Saved: models/{category_name}_opt_k_hybrid.h5")


def main():
    if os.path.exists("models"):
        shutil.rmtree("models")
    os.makedirs("models", exist_ok=True)

    print("Loading data from Hugging Face...")
    df = load_data_direct()
    if df is None or df.empty:
        print("❌ Error: Loaded DataFrame is empty.")
        return

    config     = load_config()
    categories = config['categories']

    for cat_name, asset_list in categories.items():
        if cat_name == 'benchmarks':
            continue
        train_category(cat_name, asset_list, df)

    print("\n✅ All models trained successfully.")


if __name__ == "__main__":
    main()
