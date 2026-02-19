from sklearn.svm import SVR
from stable_baselines3 import PPO
import gymnasium as gym

class HybridEngine:
    def __init__(self, trans_costs_bps):
        self.tc = trans_costs_bps / 10000.0
        self.model_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
        
    def train_svr(self, X, y):
        self.model_svr.fit(X, y)
        
    def get_ppo_action(self, state):
        # Localized import to save RAM until needed
        from stable_baselines3 import PPO
        # In a real build, we would load a pre-trained PPO zip here
        # or run a quick fine-tuning session.
        pass

    def calculate_metrics(self, returns, sofr_rate):
        excess_ret = returns - (sofr_rate / 252)
        sharpe = np.sqrt(252) * excess_ret.mean() / excess_ret.std()
        
        # Max Drawdown Calculation
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        dd = (cum_ret - peak) / peak
        return sharpe, dd.min()
