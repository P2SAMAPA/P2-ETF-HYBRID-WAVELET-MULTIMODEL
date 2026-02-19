import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os


# ---------------------------------------------------------------------------
# FEATURE BUILDER
# Defined here so app.py can do: from models.engine import MomentumEngine, build_features
# All features are lagged by >= 1 day — zero look-ahead bias.
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, target_col: str = "GLD_Ret") -> tuple:
    """
    Constructs a lag-safe feature matrix from a returns DataFrame.

    All features at row i are known at the close of day i-1, predicting
    the return on day i. This guarantees zero look-ahead bias.

    Features engineered:
      - 1, 3, 5, 10, 21-day lagged returns of the target
      - 5 and 21-day rolling realised volatility of the target (lagged 1 day)
      - All other columns in df lagged by 1 day (cross-asset / macro proxies)

    Parameters
    ----------
    df         : DataFrame of daily returns and macro levels.
    target_col : Column the SVR should predict (next-day return).

    Returns
    -------
    X            : np.ndarray (n_samples, n_features)
    y            : np.ndarray (n_samples,)  next-day target returns
    valid_index  : pd.DatetimeIndex of rows used (NaN rows dropped)
    feature_names: list[str]
    """
    feat = pd.DataFrame(index=df.index)

    # Lagged return features
    for lag in [1, 3, 5, 10, 21]:
        feat[f"ret_lag{lag}"] = df[target_col].shift(lag)

    # Rolling volatility features (lagged 1 day so known before the bar opens)
    feat["vol_5d"]  = df[target_col].rolling(5).std().shift(1)
    feat["vol_21d"] = df[target_col].rolling(21).std().shift(1)

    # Cross-asset / macro columns — all lagged 1 day
    extra_cols = [c for c in df.columns if c != target_col]
    for col in extra_cols:
        feat[f"{col}_lag1"] = df[col].shift(1)

    # Target: next-day return
    feat["__target__"] = df[target_col]

    feat.dropna(inplace=True)
    feature_names = [c for c in feat.columns if c != "__target__"]

    X = feat[feature_names].values
    y = feat["__target__"].values
    return X, y, feat.index, feature_names


# ---------------------------------------------------------------------------
# MOMENTUM ENGINE
# ---------------------------------------------------------------------------

class MomentumEngine:
    def __init__(self, c_param: float = 700.0, degree: int = 3):
        """
        High-Conviction Momentum Engine.
        SVR with 3rd-degree Polynomial Kernel, C=700.
        Wrapped in a sklearn Pipeline so live/validation data is scaled
        identically to training data via the fitted StandardScaler.
        """
        self.c_param  = c_param
        self.degree   = degree
        self.is_trained = False

        # epsilon=0.001 is appropriate for daily return magnitudes (~0.0001–0.02).
        # The previous value of 0.1 was larger than most daily returns, causing
        # the SVR to ignore most training samples and predict near-zero for everything.
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
        Trains the SVR pipeline.

        Parameters
        ----------
        X : (n_samples, n_features) — output of processor.build_feature_matrix()
        y : (n_samples,)            — next-day target returns

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
            return True
        except Exception as e:
            print(f"Training Error: {e}")
            return False

    def predict_signal(self, current_features: np.ndarray) -> float:
        """
        Predicts the next-period return for a single feature row.
        Positive → long GLD. Negative / below threshold → hold CASH.

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
        Used in backtesting to avoid slow row-by-row loops.

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
      1. Calls processor.build_feature_matrix() to get denoised, lag-safe features
      2. Trains a fresh MomentumEngine on the full dataset
      3. Saves to disk

    Parameters
    ----------
    raw_df     : Raw price + macro DataFrame from loader.py
    target_col : ETF column to predict
    save_path  : Where to persist the fitted pipeline

    Returns
    -------
    Trained MomentumEngine, or None on failure.
    """
    X, y, _, _ = build_features(raw_df, target_col=target_col)

    engine = MomentumEngine()
    success = engine.train(X, y)

    if success:
        engine.save(save_path)
        print(f"Model trained on {len(y)} samples → saved to {save_path}")
        return engine

    print("Model training failed — returning None.")
    return None
