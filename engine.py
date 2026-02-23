import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

class DeepHybridEngine:
    def __init__(self, mode="Option K", lookback=20):
        self.mode = mode
        self.lookback = lookback
        self.model = None
        self.is_trained = False
        
    def _build_parallel_model(self, n_price_feats, n_macro_feats):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Concatenate, Dropout, Attention
        
        price_in = Input(shape=(self.lookback, n_price_feats))
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(price_in)
        macro_in = Input(shape=(n_macro_feats,))
        y_feat = Dense(16, activation='relu')(macro_in)
        
        x = LSTM(64, return_sequences=True if "Option J" in self.mode else False)(x)
        if "Option J" in self.mode:
            x_att = Attention()([x, x])
            x = LSTM(32)(x_att)
        
        combined = Concatenate()([x, y_feat])
        z = Dense(32, activation='relu')(combined)
        out = Dense(1, activation='tanh')(z)
        return Model(inputs=[price_in, macro_in], outputs=out)

    def train(self, X, y):
        # Cloud models are pre-trained; this is a safety fallback
        if self.is_trained: return True
        n_price = X.shape[2]
        X_macro = np.zeros((X.shape[0], 8)) 
        self.model = self._build_parallel_model(n_price, 8)
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit([X, X_macro], y, epochs=1, batch_size=32, verbose=0)
        self.is_trained = True
        return True

    def predict_series(self, X, X_macro=None, full_index=None):
        idx = full_index if full_index is not None else range(len(X))
        if not self.is_trained or self.model is None:
            return pd.Series(0.0, index=idx)
        
        if X_macro is None:
            X_macro = np.zeros((X.shape[0], 8))
            
        # RECTIFICATION: Direct prediction without dangerous manual padding
        raw_preds = self.model.predict([X, X_macro], verbose=0).flatten()
        return pd.Series(raw_preds, index=idx)

    def save(self, filepath):
        if self.model:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)

    def load(self, filepath):
        from tensorflow.keras.models import load_model
        from tensorflow.keras import backend as K
        if os.path.exists(filepath):
            K.clear_session() # Prevent memory leaks in Streamlit
            self.model = load_model(filepath)
            self.is_trained = True

class MomentumEngine:
    def __init__(self, c_param=700.0, degree=3):
        self.is_trained = False
        self.model = Pipeline([
            ('scaler', StandardScaler()), 
            ('svr', SVR(kernel='poly', degree=degree, C=c_param))
        ])

    def train(self, X, y):
        try:
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except: return False

    def predict_series(self, X, X_macro=None, full_index=None):
        idx = full_index if full_index is not None else getattr(X, 'index', range(len(X)))
        if not self.is_trained: return pd.Series(0.0, index=idx)
        return pd.Series(self.model.predict(X), index=idx)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            self.is_trained = True

# RECTIFICATION: Added PPOEngine for Option B (Stability Locked)
class PPOEngine:
    def __init__(self, lr=0.01, epsilon=0.2):
        self.lr, self.epsilon, self.weights, self.is_trained = lr, epsilon, None, False

    def train(self, X, y):
        np.random.seed(42)  # FIXES THE FLIPPING ISSUE
        f = X.values if hasattr(X, 'values') else X
        l = y.values if hasattr(y, 'values') else y
        if self.weights is None:
            self.weights = np.random.normal(0, 0.1, f.shape[1])
        
        for _ in range(15):
            old_preds = np.dot(f, self.weights)
            ratio = np.dot(f, self.weights) / (old_preds + 1e-9)
            surr1 = ratio * (l - old_preds)
            surr2 = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * (l - old_preds)
            grad = np.dot(f.T, np.minimum(surr1, surr2)) / len(l)
            self.weights += self.lr * grad
        
        self.is_trained = True
        return True

    def predict_series(self, X, X_macro=None, full_index=None):
        idx = full_index if full_index is not None else range(len(X))
        if not self.is_trained: return pd.Series(0.0, index=idx)
        f = X.values if hasattr(X, 'values') else X
        return pd.Series(np.tanh(np.dot(f, self.weights)), index=idx)

class A2CEngine:
    def __init__(self, lr=0.01):
        self.lr, self.weights, self.is_trained = lr, None, False

    def train(self, X, y):
        np.random.seed(42)  # FIXES THE FLIPPING ISSUE
        f = X.values if hasattr(X, 'values') else X
        l = y.values if hasattr(y, 'values') else y
        if self.weights is None:
            self.weights = np.random.normal(0, 0.1, f.shape[1])
        for _ in range(10): # Increased epochs for better convergence
            preds = np.dot(f, self.weights)
            self.weights += self.lr * np.dot(f.T, l - preds) / len(l)
        self.is_trained = True
        return True

    def predict_series(self, X, X_macro=None, full_index=None):
        idx = full_index if full_index is not None else (X.index if hasattr(X, 'index') else range(len(X)))
        if not self.is_trained: return pd.Series(0.0, index=idx)
        f = X.values if hasattr(X, 'values') else X
        raw_preds = np.tanh(np.dot(f, self.weights))
        return pd.Series(raw_preds, index=idx)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.weights, filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            self.weights = joblib.load(filepath)
            self.is_trained = True

def run_bayesian_filter(series):
    if not isinstance(series, pd.Series): return series
    return series.rolling(window=5, min_periods=1).mean()
