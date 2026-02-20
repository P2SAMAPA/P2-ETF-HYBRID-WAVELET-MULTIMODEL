import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

class RegimeHMM:
    def __init__(self, n_states=3):
        # We use 3 states: 
        # State 0: Low Vol/Expansion, State 1: High Vol/Contraction, State 2: Transition
        self.model = GaussianHMM(
            n_components=n_states, 
            covariance_type="full", 
            n_iter=1000,
            random_state=42
        )
        self.state_map = {}

    def train_and_assign(self, df, assets):
        """
        df: The dataframe from load_raw_data()
        assets: List of tickers like ['GLD', 'TLT', etc.]
        """
        # 1. Define the 5 Pillars of Macro Context
        macro_cols = ["VIX", "DXY", "T10Y2Y", "IG_SPREAD", "HY_SPREAD"]
        
        # 2. Prepare Features: We use daily changes (diff) to identify shifts
        # Dropping NaNs is critical for HMM stability
        macro_data = df[macro_cols].diff().dropna()
        asset_rets = df[[f"{a}_Ret" for a in assets]].loc[macro_data.index]
        
        # 3. Fit the HMM
        self.model.fit(macro_data)
        
        # 4. Labeling the States
        # We look at which asset outperformed in which hidden state
        hidden_states = self.model.predict(macro_data)
        state_results = pd.DataFrame({'State': hidden_states}, index=macro_data.index)
        combined = state_results.join(asset_rets)
        
        for s in range(self.model.n_components):
            # Calculate mean return per asset in this specific state
            avg_rets = combined[combined['State'] == s].drop(columns='State').mean()
            # Map the State ID to the winning asset ticker
            best_asset = avg_rets.idxmax().replace("_Ret", "")
            self.state_map[s] = best_asset

    def predict_best_asset(self, current_macro_sample):
        """
        Predicts the best asset based on the current macro state.
        current_macro_sample: 2D array of the latest macro diffs
        """
        try:
            state = self.model.predict(current_macro_sample)[0]
            return self.state_map.get(state, "CASH")
        except:
            return "CASH"
