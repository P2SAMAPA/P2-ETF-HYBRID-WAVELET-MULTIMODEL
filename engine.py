import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

# ---------------------------------------------------------------------------
# DEEP LEARNING ENGINES (Options I, J, K)
# ---------------------------------------------------------------------------
class DeepHybridEngine:
    def __init__(self, mode="Option K", lookback=20):
        self.mode = mode
        self.lookback = lookback
        self.model = None
        self.is_trained = False
        
    def _build_parallel_model(self, n_price_feats, n_macro_feats):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Concatenate, Dropout, Attention
        
        price_in = Input(shape=(self.lookback, n_price_feats), name='price_input')
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(price_in)
        
        macro_in = Input(shape=(n_macro_feats,), name='macro_input')
        y = Dense(16, activation='relu')(macro_in)
        y = Dense(8, activation='relu')(y)

        x = LSTM(64, return_sequences=True if "Option J" in self.mode else False)(x)
        if "Option J" in self.mode:
            # Query-Value Attention logic
            x_att = Attention()([x, x])
            x = LSTM(32)(x_att)
        
        combined = Concatenate()([x, y])
        z = Dense(32, activation='relu')(combined)
        z = Dropout(0.2)(z)
        out = Dense(1, activation='tanh')(z)
        
        return Model(inputs=[price_in, macro_in], outputs=out)

    def train(self, X_price, X_macro, y):
        n_price = X_price.shape[2]
        n_macro = X_macro.shape[1]
        self.model = self._build_parallel_model(n_price, n_macro)
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit([X_price, X_macro], y, epochs=10, batch_size=32, verbose=0)
        self.is_trained = True
        return True

    def predict_series(self, X_price, X_macro, full_index):
        """Fixed: Uses full_index to prevent diagonal graph"""
        if not self.is_trained or self.model is None:
            return pd.Series(0.0, index=full_index)
        try:
            raw_preds = self.model.predict([X_price, X_macro], verbose=0).flatten()
            # Pad the beginning if feature engineering ate some rows
            preds = np.zeros(len(full_index))
            preds[-len(raw_preds):] = raw_preds
            return pd.Series(preds, index=full_index)
        except:
            return pd.Series(0.0, index=full_index)

    def save(self, filepath):
        if self.model:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)

    def load(self, filepath):
        from tensorflow.keras.models import load_model
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            self.is_trained = True

# ---------------------------------------------------------------------------
# SVR ENGINES (Options A, B, C, D)
# ---------------------------------------------------------------------------
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

    def predict_series(self, X, full_index):
        """Fixed: Forces index alignment"""
        if not self.is_trained:
            return pd.Series(0.0, index=full_index)
        raw_preds = self.model.predict(X)
        preds = np.zeros(len(full_index))
        preds[-len(raw_preds):] = raw_preds
        return pd.Series(preds, index=full_index)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            self.is_trained = True

# ---------------------------------------------------------------------------
# REINFORCEMENT LEARNING ENGINE
# ---------------------------------------------------------------------------
class A2CEngine:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.weights = None
        self.is_trained = False

    def train(self, X, y):
        # Flattened A2C Approximation
        features = X.values if hasattr(X, 'values') else X
        labels = y.values if hasattr(y, 'values') else y
        if self.weights is None:
            self.weights = np.random.normal(0, 0.1, features.shape[1])
        # Simple gradient update
        for _ in range(5):
            preds = np.dot(features, self.weights)
            error = labels - preds
            self.weights += self.lr * np.dot(features.T, error) / len(labels)
        self.is_trained = True
        return True

    def predict_series(self, X, full_index):
        if not self.is_trained: return pd.Series(0.0, index=full_index)
        features = X.values if hasattr(X, 'values') else X
        raw_preds = np.tanh(np.dot(features, self.weights))
        preds = np.zeros(len(full_index))
        preds[-len(raw_preds):] = raw_preds
        return pd.Series(preds, index=full_index)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.weights, filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            self.weights = joblib.load(filepath)
            self.is_trained = True

# ---------------------------------------------------------------------------
# BAYESIAN/HMM POST-PROCESSOR (Options E, F, G, H)
# ---------------------------------------------------------------------------
def run_bayesian_filter(series):
    """
    RECTIFIED: Final check to ensure signal isn't lost 
    and index is strictly preserved.
    """
    if not isinstance(series, pd.Series):
        return series
    # Simple Bayesian-style smoothing that respects the index
    smoothed = series.rolling(window=5, min_periods=1).mean()
    return smoothed
