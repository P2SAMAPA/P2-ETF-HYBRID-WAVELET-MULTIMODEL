import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib  # FIXED: was 'jobpath' which is not a real library
import os


class MomentumEngine:
    def __init__(self, c_param=700.0, degree=3):
        """
        Initializes the High-Conviction Momentum Engine.
        SVR with a 3rd-degree Polynomial Kernel, C=700 for high-conviction
        trend-following. Wrapped in a Pipeline so validation/live data is
        scaled identically to training data.
        """
        self.c_param = c_param
        self.degree = degree
        self.is_trained = False

        # FIXED: epsilon=0.001 is appropriate for daily return magnitudes
        # (0.0001–0.02 range). The previous value of 0.1 was far too large
        # and would have caused the SVR to ignore most training samples,
        # producing near-zero predictions for everything.
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(
                kernel='poly',
                degree=self.degree,
                C=self.c_param,
                epsilon=0.001,   # FIXED: was 0.1 — too large for daily returns
                coef0=1.0,
                gamma='scale'
            ))
        ])

    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Trains the SVR model on macro/technical features (X) to predict
        next-period asset returns (y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix — e.g. lagged returns, VIX, DXY, T10Y2Y spread.
        y : np.ndarray, shape (n_samples,)
            Target next-period returns (e.g. GLD_Ret shifted by -1 so the
            model is predicting tomorrow's return from today's features).

        Returns
        -------
        bool : True if training succeeded, False otherwise.
        """
        if len(X) != len(y):
            print(f"Training Error: X rows ({len(X)}) != y rows ({len(y)})")
            return False
        if len(X) < 50:
            print("Training Error: Need at least 50 samples to train.")
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
        Generates a predicted next-period return for live signal generation.

        Parameters
        ----------
        current_features : np.ndarray, shape (1, n_features)
            Today's feature row. Must be shaped (1, n) before passing in.

        Returns
        -------
        float : Predicted return. Positive → go long GLD. Negative → hold CASH.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        if current_features.ndim == 1:
            current_features = current_features.reshape(1, -1)
        prediction = self.model.predict(current_features)
        return float(prediction[0])

    def predict_series(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predicted returns for an entire feature matrix (used in
        backtesting so app.py can generate the full signal series without
        a slow Python loop calling predict_signal row by row).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) of predicted returns.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self.model.predict(X)

    def save(self, filepath: str = "models/svr_momentum_poly.pkl") -> bool:
        """Persists the trained pipeline to disk."""
        if not self.is_trained:
            print("Save skipped: model is not trained yet.")
            return False
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.model, filepath)
            return True
        except Exception as e:
            print(f"Save Error: {e}")
            return False

    @classmethod
    def load(cls, filepath: str = "models/svr_momentum_poly.pkl") -> "MomentumEngine":
        """
        Loads a previously saved Pipeline from disk and wraps it back into
        a MomentumEngine instance ready for prediction.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model found at: {filepath}")
        instance = cls.__new__(cls)
        instance.model = joblib.load(filepath)
        instance.is_trained = True
        return instance


# ---------------------------------------------------------------------------
# PRODUCTION UTILITY
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, target_col: str = "GLD_Ret") -> tuple:
    """
    Constructs a lagged feature matrix from a return DataFrame so that
    there is ZERO look-ahead bias: all features at row i are known at
    the close of day i-1, predicting the return on day i.

    Features engineered:
      - 1, 3, 5, 10, 21-day lagged returns of the target
      - 5 and 21-day rolling volatility of the target (lagged 1 day)
      - Any additional columns already in df (e.g. VIX, DXY) — also lagged 1

    Returns
    -------
    X : np.ndarray  (no NaN rows)
    y : np.ndarray  (aligned next-day returns)
    valid_index : pd.DatetimeIndex of the rows used
    feature_names : list[str]
    """
    feat = pd.DataFrame(index=df.index)

    # Lagged return features — all shifted so they are known before the return
    for lag in [1, 3, 5, 10, 21]:
        feat[f"ret_lag{lag}"] = df[target_col].shift(lag)

    # Rolling volatility features (also lagged to avoid look-ahead)
    feat["vol_5d"]  = df[target_col].rolling(5).std().shift(1)
    feat["vol_21d"] = df[target_col].rolling(21).std().shift(1)

    # Any extra columns in df (e.g. SPY_Ret, AGG_Ret as macro proxies)
    extra_cols = [c for c in df.columns if c != target_col]
    for col in extra_cols:
        feat[f"{col}_lag1"] = df[col].shift(1)

    # Target: next-day return (what we are predicting)
    feat["__target__"] = df[target_col]

    feat.dropna(inplace=True)
    feature_names = [c for c in feat.columns if c != "__target__"]

    X = feat[feature_names].values
    y = feat["__target__"].values
    return X, y, feat.index, feature_names


def update_model_checkpoint(
    df: pd.DataFrame,
    target_col: str = "GLD_Ret",
    save_path: str = "models/svr_momentum_poly.pkl"
) -> "MomentumEngine | None":
    """
    Helper to retrain the SVR engine with the latest market data and
    optionally persist it to disk.

    Parameters
    ----------
    df         : DataFrame containing return columns.
    target_col : The column the SVR should predict.
    save_path  : Where to save the fitted Pipeline.

    Returns
    -------
    Trained MomentumEngine, or None on failure.
    """
    X, y, _, _ = build_features(df, target_col=target_col)

    engine = MomentumEngine()
    success = engine.train(X, y)

    if success:
        engine.save(save_path)
        print(f"Model trained on {len(y)} samples and saved to {save_path}.")
        return engine

    print("Model training failed — returning None.")
    return None
