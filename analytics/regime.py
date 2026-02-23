import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import scipy.stats as stats

class RegimeHMM:
    def __init__(self, n_states=3):
        # Increased n_iter for better convergence on complex macro shifts
        self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=2000, random_state=42)
        self.state_map = {}

    def train_and_assign(self, df, assets):
        macro_cols = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]
        macro_data = df[macro_cols].diff().dropna()
        asset_rets = df[[f"{a}_Ret" for a in assets]].loc[macro_data.index]
        
        self.model.fit(macro_data)
        states = self.model.predict(macro_data)
        
        combined = pd.DataFrame({'State': states}, index=macro_data.index).join(asset_rets)
        for s in range(self.model.n_components):
            avg_rets = combined[combined['State'] == s].drop(columns='State').mean()
            # AGGRESSIVE: If the best asset is negative, still pick the 'least bad' 
            # instead of defaulting to a state that might map to CASH later.
            self.state_map[s] = avg_rets.idxmax().replace("_Ret", "")

    def predict_best_asset(self, current_macro_sample):
        try:
            state = self.model.predict(current_macro_sample)[0]
            return self.state_map.get(state, "SPY") # Aggressive default to SPY instead of CASH
        except: 
            return "SPY"

class BayesianFilter:
    """Aggressive Trend confidence filter"""
    def get_confidence(self, series):
        # RECTIFIED: Reduced data window requirement to be more reactive
        if len(series) < 10: return 1.0
        
        returns = series.pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        
        if sigma == 0: return 1.0

        # RECTIFIED: AGGRESSIVE BAYESIAN MATH
        # We remove the sqrt(n) dampener to make the 'mu' (trend) more dominant.
        # We add a 0.1 'boost' to the loc to favor taking a position.
        aggressive_mu = mu + (0.05 * sigma) 
        prob = 1 - stats.norm.cdf(0, loc=aggressive_mu, scale=sigma)
        
        # Clamp the probability so it doesn't zero out signals too easily
        return max(0.1, prob)
