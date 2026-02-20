import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

# NEW: Import for Deep Learning Engines
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Concatenate, Dropout, Flatten

# ---------------------------------------------------------------------------
# DEEP LEARNING ENGINES (I, J, K)
# ---------------------------------------------------------------------------

class DeepHybridEngine:
    def __init__(self, mode="Option K", lookback=20):
        """
        Deep Learning Engine for Spatial-Temporal Feature Extraction.
        Modes: 
        - Option I: Wavelet-CNN-LSTM (Single Stream)
        - Option J: SVR-CNN-LSTM (Residual Modeling)
        - Option K: Parallel-Dual-Stream (Price + Macro/Credit Gates)
        """
        self.mode = mode
        self.lookback = lookback
        self.model = None
        self.is_trained = False

    def _build_parallel_model(self, n_price_feats, n_macro_feats):
        """Architecture for Option K: Dual-Input Functional API"""
        # Stream 1: Price Action (CNN-LSTM)
        price_in = Input(shape=(self.lookback, n_price_feats), name='price_input')
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(price_in)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)

        # Stream 2: Macro/Credit Regime (Dense)
        # Focuses on HY Spreads, VIX, DXY
        macro_in = Input(shape=(n_macro_feats,), name='macro_input')
        y = Dense(16, activation='relu')(macro_in)
        y = Dense(8, activation='relu')(y)

        # Fusion Layer: Combining technical shapes with credit risk context
        merged = Concatenate()([x, y])
        z = Dense(16, activation='relu')(merged)
        output = Dense(1, activation='linear')(z)

        model = Model(inputs=[price_in, macro_in], outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_price, y, X_macro=None):
        """Trains the deep engine based on the selected architecture."""
        try:
            if self.mode == "Option K":
                self.model = self._build_parallel_model(X_price.shape[2], X_macro.shape[1])
                self.model.fit([X_price, X_macro], y, epochs=10, batch_size=32, verbose=0)
            else:
                # Logic for Option I & J (Single Input)
                price_in = Input(shape=(self.lookback, X_price.shape[2]))
                x = Conv1D(32, 3, activation='relu')(price_in)
                x = LSTM(50)(x)
                out = Dense(1)(x)
                self.model = Model(inputs=price_in, outputs=out)
                self.model.compile(optimizer='adam', loss='mse')
                self.model.fit(X_price, y, epochs=10, verbose=0)
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"DL Training Error: {e}")
            return False

    def predict_series(self, X_price, X_macro=None):
        if not self.is_trained: return np.zeros(len(X_price))
        
        if self.mode == "Option K":
            return self.model.predict([X_price, X_macro]).flatten()
        return self.model.predict(X_price).flatten()

    def save(self, filepath):
        if self.model: self.model.save(filepath)

# ---------------------------------------------------------------------------
# MOMENTUM ENGINE (Existing - Untouched)
# ---------------------------------------------------------------------------
class MomentumEngine:
    def __init__(self, c_param: float = 700.0, degree: int = 3):
        self.c_param = c_param
        self.degree = degree
        self.is_trained = False
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='poly', degree=self.degree, C=self.c_param, epsilon=0.001, coef0=1.0, gamma='scale'))
        ])

    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        if len(X) != len(y): return False
        try:
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except Exception as e:
            return False

    def predict_series(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained: return np.zeros(len(X))
        return self.model.predict(X)

    def save(self, filepath: str = "models/svr_momentum_poly.pkl") -> bool:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        return True

# ---------------------------------------------------------------------------
# A2C ENGINE (Existing - Untouched)
# ---------------------------------------------------------------------------
class A2CEngine:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.weights = None 

    def train(self, X, y):
        features = X.values if hasattr(X, 'values') else X
        labels = y.values if hasattr(y, 'values') else y
        if self.weights is None:
            self.weights = np.random.normal(0, 0.1, features.shape[1])
        gradient = np.dot(features.T, labels)
        self.weights += self.lr * gradient

    def predict_series(self, X):
        features = X.values if hasattr(X, 'values') else X
        if self.weights is None:
            self.weights = np.random.normal(0, 0.1, features.shape[1])
        return np.dot(features, self.weights)
# ---------------------------------------------------------------------------
# PRODUCTION UTILITY
# ---------------------------------------------------------------------------
def update_model_checkpoint(
    raw_df: pd.DataFrame,
    target_col: str = "GLD",
    save_path: str = "models/svr_momentum_poly.pkl"
) -> "MomentumEngine | None":
    """
    End-to-end retraining helper:
      1. Calls build_features() (processor.py) to get denoised, lag-safe features
      2. Trains a fresh MomentumEngine
      3. Saves to disk
    Parameters
    ----------
    raw_df : Raw price + macro DataFrame from loader.py
    target_col : ETF column to predict (e.g. "GLD")
    save_path : Where to persist the fitted pipeline
    """
    from data.processor import build_feature_matrix as build_features
    X, y, _, _ = build_features(raw_df, target_col=target_col)
    engine = MomentumEngine()
    success = engine.train(X, y)
    if success:
        engine.save(save_path)
        return engine
    print("Model training failed — returning None.")
    return None
