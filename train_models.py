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

from data.loader import load_raw_data
from data.processor import build_feature_matrix, get_canonical_feature_names
from engine import DeepHybridEngine


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def train_category(category_name, asset_list, raw_df, lookback=20):
    """Train all three models for a given category."""
    print(f"\n🔄 Processing category: {category_name} with {len(asset_list)} assets...")

    # FIX: build canonical feature name list ONCE for this universe.
    # Every ETF in this category will produce a feature matrix aligned to
    # exactly these columns — guaranteeing identical width for concatenation.
    canonical_names = get_canonical_feature_names(
        raw_df=raw_df,
        feature_symbols=asset_list,
        include_options=True,
        include_volume=True,
        market_proxy="SPY",
    )
    # Remove the cross-asset return slot for target itself
    # (it will be filled 0 by alignment — harmless but cleaner)
    n_features = len(canonical_names)
    print(f"  Canonical feature width: {n_features} columns")

    all_X_3d   = []
    all_y_3d   = []
    all_X_macro = []

    for ticker in asset_list:
        try:
            X, y, idx, feat_names = build_feature_matrix(
                raw_df,
                target_col=ticker,
                feature_symbols=asset_list,
                canonical_names=canonical_names,   # FIX: align to canonical list
            )

            n_samples = len(X) - lookback + 1
            if n_samples <= 0:
                print(f"  ⚠️  {ticker}: Not enough data for window {lookback}. Skipping.")
                continue

            # FIX: n_features now guaranteed consistent across all tickers
            X_ticker_3d = np.zeros((n_samples, lookback, n_features))
            y_ticker_3d = np.zeros(n_samples)

            for i in range(lookback, len(X) + 1):
                X_ticker_3d[i - lookback] = X[i - lookback:i]
                y_ticker_3d[i - lookback] = y[i - 1]

            all_X_3d.append(X_ticker_3d)
            all_y_3d.append(y_ticker_3d)

            # Macro features for Option K
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

            print(f"  ✅ {ticker}: Prepared {n_samples} samples "
                  f"({n_features} features).")

        except Exception as e:
            print(f"  ❌ Skipping {ticker} due to error: {e}")
            import traceback; traceback.print_exc()

    if not all_X_3d:
        print(f"❌ No training data for category {category_name}. Skipping.")
        return

    # Concatenate — all arrays now have identical shape[2] = n_features
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

    # Option I
    print(f"\n🚀 Training Option I (CNN) for {category_name}...")
    model_i = DeepHybridEngine(mode="Option I")
    if model_i.train(X_train, y_train):
        model_i.save(f"models/{category_name}_opt_i_cnn.h5")
        print(f"  💾 Saved: models/{category_name}_opt_i_cnn.h5")

    # Option J
    print(f"\n🚀 Training Option J (Attention-CNN-LSTM) for {category_name}...")
    model_j = DeepHybridEngine(mode="Option J")
    if model_j.train(X_train, y_train):
        model_j.save(f"models/{category_name}_opt_j_cnn_lstm.h5")
        print(f"  💾 Saved: models/{category_name}_opt_j_cnn_lstm.h5")

    # Option K
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
    df, _ = load_raw_data(force_sync=False)
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
