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
            query_value_attention_seq = Attention()([x, x])
            x = LSTM(32)(query_value_attention_seq)
            
        x = Dropout(0.2)(x)
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
            if "Option K" in self.mode:
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
        from tensorflow.keras.models import load_model
        
        file_map = {"Option I": "opt_i_cnn.h5", "Option J": "opt_j_cnn_lstm.h5", "Option K": "opt_k_hybrid.h5"}
        key = "Option I" if "Option I" in self.mode else "Option J" if "Option J" in self.mode else "Option K"
        model_file = file_map.get(key)
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [os.path.join(base_dir, "models", model_file), os.path.join(base_dir, model_file)]
        model_path = next((p for p in possible_paths if os.path.exists(p) and os.path.getsize(p) > 2048), None)
        
        if model_path:
            try:
                from tensorflow.keras import backend as K
                K.clear_session()
                current_model = load_model(model_path, compile=False)
                if "Option K" in self.mode and X_macro is not None:
                    return current_model.predict([X_price, X_macro], verbose=0).flatten()
                return current_model.predict(X_price, verbose=0).flatten()
            except: pass
        return np.zeros(len(X_price))

    def save(self, filepath):
        if self.model:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)

# ---------------------------------------------------------------------------
# SVR & RL ENGINES
# ---------------------------------------------------------------------------
class MomentumEngine:
    def __init__(self, c_param=700.0, degree=3):
        self.is_trained = False
        self.model = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='poly', degree=degree, C=c_param))])

    def train(self, X, y):
        try:
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except: return False

    def predict_series(self, X):
        return self.model.predict(X) if self.is_trained else np.zeros(len(X))

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

class A2CEngine:
    def __init__(self, lr=0.01):
        self.lr, self.weights = lr, None 

    def train(self, X, y):
        f, l = (X.values if hasattr(X, 'values') else X), (y.values if hasattr(y, 'values') else y)
        if self.weights is None: self.weights = np.random.normal(0, 0.1, f.shape[1])
        self.weights += self.lr * np.dot(f.T, l)

    def predict_series(self, X):
        f = X.values if hasattr(X, 'values') else X
        if self.weights is None: self.weights = np.random.normal(0, 0.1, f.shape[1])
        return np.dot(f, self.weights)
