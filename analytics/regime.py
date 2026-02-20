import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

class RegimeHMM:
    def __init__(self, n_states=3):
        self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
        self.state_map = {}

    def train_and_assign(self, df, assets):
        # 5-Pillar Macro Context
        macro_cols = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]
        macro_data = df[macro_cols].diff().dropna()
        asset_rets = df[[f"{a}_Ret" for a in assets]].loc[macro_data.index]
        
        self.model.fit(macro_data)
        states = self.model.predict(macro_data)
        
        combined = pd.DataFrame({'State': states}, index=macro_data.index).join(asset_rets)
        for s in range(self.model.n_components):
            avg_rets = combined[combined['State'] == s].drop(columns='State').mean()
            self.state_map[s] = avg_rets.idxmax().replace("_Ret", "")

    def predict_best_asset(self, current_macro_sample):
        try:
            state = self.model.predict(current_macro_sample)[0]
            return self.state_map.get(state, "CASH")
        except: return "CASH"

class BayesianFilter:
    """Simplified BSTS-style trend confidence filter"""
    def get_confidence(self, series):
        # Calculates probability that current trend is structural rather than noise
        # Logic: Signal-to-Noise ratio + Bayesian update
        if len(series) < 20: return 1.0
        returns = series.pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        # Probability trend is positive (Simplified Bayesian approach)
        import scipy.stats as stats
        prob = 1 - stats.norm.cdf(0, loc=mu, scale=sigma/np.sqrt(len(returns)))
        return prob
