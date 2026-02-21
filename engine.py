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
        self.mode = mode
        self.lookback = lookback
        self.model = None
        self.is_trained = False
        
    def _build_parallel_model(self, n_price_feats, n_macro_feats):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Concatenate, Dropout
        
        price_in = Input(shape=(self.lookback, n_price_feats), name='price_input')
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(price_in)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)

        macro_in = Input(shape=(n_macro_feats,), name='macro_input')
        y = Dense(16, activation='relu')(macro_in)
        y = Dense(8, activation='relu')(y)

        merged = Concatenate()([x, y])
        z = Dense(16, activation='relu')(merged)
        output = Dense(1, activation='linear')(z)

        model = Model(inputs=[price_in, macro_in], outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_price, y, X_macro=None):
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
        import os
        import numpy as np
        from tensorflow.keras.models import load_model
        
        file_map = {
            "Option I": "opt_i_cnn.h5",
            "Option J": "opt_j_cnn_lstm.h5",
            "Option K": "opt_k_hybrid.h5"
        }
        
        if self.mode in file_map:
            model_file = file_map[self.mode]
            # RECTIFIED PATHING: Ensure absolute path to 'models' folder
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", model_file)
            
            if os.path.exists(model_path):
                try:
                    current_model = load_model(model_path, compile=False)
                    if self.mode == "Option K" and X_macro is not None:
                        preds = current_model.predict([X_price, X_macro], verbose=0).flatten()
                    else:
                        preds = current_model.predict(X_price, verbose=0).flatten()
                    return preds
                except Exception as e:
                    print(f"❌ Load Error for {self.mode}: {e}")
            else:
                print(f"⚠️ Missing File: {model_path}")

        return np.zeros(len(X_price))

    def save(self, filepath):
        if self.model: self.model.save(filepath)

# ---------------------------------------------------------------------------
# MOMENTUM ENGINE (UNTOUCHED)
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
# A2C ENGINE (UNTOUCHED)
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

def update_model_checkpoint(raw_df: pd.DataFrame, target_col: str = "GLD", save_path: str = "models/svr_momentum_poly.pkl") -> "MomentumEngine | None":
    from data.processor import build_feature_matrix as build_features
    X, y, _, _ = build_features(raw_df, target_col=target_col)
    engine = MomentumEngine()
    success = engine.train(X, y)
    if success:
        engine.save(save_path)
        return engine
    return None
