import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

# ---------------------------------------------------------------------------
# DEEP LEARNING ENGINES (I, J, K)
# ---------------------------------------------------------------------------

class DeepHybridEngine:
    def __init__(self, mode="Option K", lookback=20):
        """
        Deep Learning Engine for Spatial-Temporal Feature Extraction.
        """
        self.mode = mode
        self.lookback = lookback
        self.model = None
        self.is_trained = False
        
        # Internal imports to keep the global namespace clean
        import tensorflow as tf

    def _build_parallel_model(self, n_price_feats, n_macro_feats):
        """Architecture for Option K: Dual-Input Functional API"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Concatenate, Dropout
        
        # Stream 1: Price Action (CNN-LSTM)
        price_in = Input(shape=(self.lookback, n_price_feats), name='price_input')
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(price_in)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)

        # Stream 2: Macro/Credit Regime (Dense)
        macro_in = Input(shape=(n_macro_feats,), name='macro_input')
        y = Dense(16, activation='relu')(macro_in)
        y = Dense(8, activation='relu')(y)

        # Fusion Layer
        merged = Concatenate()([x, y])
        z = Dense(16, activation='relu')(merged)
        output = Dense(1, activation='linear')(z)

        model = Model(inputs=[price_in, macro_in], outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_price, y, X_macro=None):
        """Trains the deep engine based on the selected architecture."""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense
        try:
            if self.mode == "Option K":
                self.model = self._build_parallel_model(X_price.shape[2], X_macro.shape[1])
                self.model.fit([X_price, X_macro], y, epochs=10, batch_size=32, verbose=0)
            else:
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
        """Standardized Prediction Logic for Options I, J, K"""
        import streamlit as st
        
        # UPDATED FILE MAP: Matching your automated filenames
        file_map = {
            "Option I": "opt_i_cnn.h5",
            "Option J": "opt_j_cnn_lstm.h5", # Updated to match automated name
            "Option K": "opt_k_hybrid.h5"    # Updated to match automated name
        }
        
        # 1. Load Model if needed
        if self.mode in file_map:
            model_path = os.path.join("models", file_map[self.mode])
            
            # DEBUG LOG: This will show up in your Hugging Face Logs
            if not os.path.exists(model_path):
                print(f"⚠️ Path not found: {model_path}. Fallback to SVR likely.")
            
            if os.path.exists(model_path) and self.model is None:
                from tensorflow.keras.models import load_model
                try:
                    self.model = load_model(model_path, compile=False)
                    self.is_trained = True
                    print(f"✅ Successfully loaded {self.mode} from {model_path}")
                except Exception as e:
                    print(f"❌ Error loading {model_path}: {e}")
                    return np.zeros(len(X_price))

        # 2. Ensure we have a model to use
        if self.model is None:
            return np.zeros(len(X_price))

        # 3. AUTO-RESHAPE: Ensure input is 3D (Samples, Lookback, Features)
        try:
            if len(X_price.shape) == 2:
                # Sliding window conversion
                X_3d = np.array([X_price[i-self.lookback:i] for i in range(self.lookback, len(X_price)+1)])
                # Pad start with zeros to maintain series length alignment
                padding = np.zeros((self.lookback-1, self.lookback, X_price.shape[1]))
                X_final = np.vstack([padding, X_3d])
            else:
                X_final = X_price
        except Exception:
            X_final = X_price # Fallback

        # 4. Execute Prediction based on mode
        try:
            if self.mode == "Option K":
                if X_macro is None: 
                    return np.zeros(len(X_price))
                return self.model.predict([X_final, X_macro], verbose=0).flatten()
            
            return self.model.predict(X_final, verbose=0).flatten()
        except Exception as e:
            print(f"Prediction Error in {self.mode}: {e}")
            return np.zeros(len(X_price))

    def save(self, filepath):
        if self.model: self.model.save(filepath)

# ---------------------------------------------------------------------------
# MOMENTUM ENGINE (Untouched)
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
# A2C ENGINE (Untouched)
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
# PRODUCTION UTILITY (Untouched)
# ---------------------------------------------------------------------------
def update_model_checkpoint(
    raw_df: pd.DataFrame,
    target_col: str = "GLD",
    save_path: str = "models/svr_momentum_poly.pkl"
) -> "MomentumEngine | None":
    from data.processor import build_feature_matrix as build_features
    X, y, _, _ = build_features(raw_df, target_col=target_col)
    engine = MomentumEngine()
    success = engine.train(X, y)
    if success:
        engine.save(save_path)
        return engine
    print("Model training failed — returning None.")
    return None
