import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
# ---------------------------------------------------------------------------
# MOMENTUM ENGINE
# ---------------------------------------------------------------------------
class MomentumEngine:
    def __init__(self, c_param: float = 700.0, degree: int = 3):
        """
        High-Conviction Momentum Engine.
        SVR with 3rd-degree Polynomial Kernel, C=700 for aggressive
        trend-following. Wrapped in a sklearn Pipeline so validation
        and live data are scaled identically to training data.
        """
        self.c_param = c_param
        self.degree = degree
        self.is_trained = False
        # epsilon=0.001 is appropriate for daily return magnitudes (~0.0001-0.02).
        # Original value of 0.1 was larger than most daily returns, causing the
        # SVR to ignore most training samples and predict near-zero everywhere.
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(
                kernel='poly',
                degree=self.degree,
                C=self.c_param,
                epsilon=0.001,
                coef0=1.0,
                gamma='scale'
            ))
        ])
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Trains the SVR pipeline on the feature matrix from build_features().
        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,) — next-day target returns
        Returns True on success, False on failure.
        """
        if len(X) != len(y):
            print(f"Training Error: X rows ({len(X)}) != y rows ({len(y)})")
            return False
        if len(X) < 50:
            print(f"Training Error: only {len(X)} samples — need at least 50.")
            return False
        try:
            self.model.fit(X, y)
            self.is_trained = True
            print(f"SVR trained on {len(X)} samples.")
            return True
        except Exception as e:
            print(f"Training Error: {e}")
            return False
    def predict_signal(self, current_features: np.ndarray) -> float:
        """
        Predicts next-period return for a single feature row.
        Positive → long GLD. At or below threshold → hold CASH.
        Parameters
        ----------
        current_features : (1, n_features) or (n_features,)
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        if current_features.ndim == 1:
            current_features = current_features.reshape(1, -1)
        return float(self.model.predict(current_features)[0])
    def predict_series(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorised prediction over an entire feature matrix.
        Used in backtesting to score the full OOS period in one call.
        Parameters
        ----------
        X : (n_samples, n_features)
        Returns
        -------
        np.ndarray (n_samples,) of predicted returns.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self.model.predict(X)
    def save(self, filepath: str = "models/svr_momentum_poly.pkl") -> bool:
        """Persists the trained pipeline to disk via joblib."""
        if not self.is_trained:
            print("Save skipped: model is not trained.")
            return False
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Save Error: {e}")
            return False
    @classmethod
    def load(cls, filepath: str = "models/svr_momentum_poly.pkl") -> "MomentumEngine":
        """Loads a saved Pipeline from disk into a MomentumEngine instance."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model at: {filepath}")
        instance = cls.__new__(cls)
        instance.model = joblib.load(filepath)
        instance.is_trained = True
        return instance
import numpy as np

class A2CEngine:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.gamma = 0.95  # Discount factor for future rewards
        # Simplified A2C weights for high-frequency signal processing
        self.weights = None 

    def predict(self, features):
        # A2C Advantage calculation
        # If weights aren't initialized, we start with a neutral macro-bias
        if self.weights is None:
            self.weights = np.random.normal(0, 1, features.shape[1])
        
        # Calculate policy (Actor) and state-value (Critic)
        # Higher values indicate a stronger 'Advantage' over Cash
        advantage_scores = np.dot(features, self.weights)
        return advantage_scores

    def train_step(self, features, rewards):
        # This simulates the synchronous update of Actor and Critic
        # Higher rewards boost the weights of the features that led to them
        self.weights += self.lr * np.dot(features.T, rewards)
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
