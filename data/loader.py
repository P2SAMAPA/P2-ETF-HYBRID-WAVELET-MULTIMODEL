import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download
import io

class FeatureLoader:
    def __init__(self, fred_key, hf_token, repo_id):
        self.fred = Fred(api_key=fred_key)
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.tickers = ["TLT", "TBT", "VNQ", "SLV", "GLD", "SPY", "AGG"] # Added benchmarks
        self.macros = {"DXY": "DTWEXBGS", "VIX": "VIXCLS", "T10Y2Y": "T10Y2Y", "SOFR": "SOFR"}

    def sync_data(self, force_full_reseed=False):
        try:
            # Download the existing file
            path = hf_hub_download(repo_id=self.repo_id, filename="master_data.parquet", 
                                   repo_type="dataset", token=self.hf_token)
            master_df = pd.read_parquet(path)
            # Check if history is missing (e.g., starts later than 2008)
            if master_df.index.min().year > 2008: force_full_reseed = True
            last_date = master_df.index.max()
        except Exception:
            master_df = pd.DataFrame()
            last_date = pd.Timestamp("2008-01-01")
            force_full_reseed = True

        # logic to prevent future-dated data
        today = pd.Timestamp.now().normalize()
        if not force_full_reseed and last_date >= (today - pd.Timedelta(days=1)):
            return "Already Refreshed"

        try:
            start_fetch = pd.Timestamp("2008-01-01") if force_full_reseed else last_date
            # Ensure multi_level_index=False to prevent tuple errors
            raw_data = yf.download(self.tickers, start=start_fetch, multi_level_index=False)
            
            # 1. Flatten Tuple Columns immediately
            # yfinance often returns "Price_Ticker" strings when multi_level_index=False
            data = raw_data[[c for c in raw_data.columns if 'Close' in c]].copy()
            data.columns = [c.replace('Close_', '').strip() for c in data.columns]
            
            # 2. Fetch Macro Data
            macro_frames = []
            for name, fred_id in self.macros.items():
                s = self.fred.get_series(fred_id, start_fetch)
                macro_frames.append(pd.Series(s, name=name))
            
            macro_df = pd.concat(macro_frames, axis=1).ffill()
            
            # 3. Combine and filter for only completed days
            combined = pd.concat([data, macro_df], axis=1).ffill().dropna()
            combined = combined[combined.index < today] # Only dates strictly before today

            # Merge with master or replace if reseeding
            final_df = combined if force_full_reseed else pd.concat([master_df, combined])
            final_df = final_df[~final_df.index.duplicated(keep='last')].sort_index()

            # Upload back to HF
            buf = io.BytesIO()
            final_df.to_parquet(buf)
            HfApi().upload_file(path_or_fileobj=buf.getvalue(), path_in_repo="master_data.parquet", 
                                repo_id=self.repo_id, repo_type="dataset", token=self.hf_token)
            
            return f"Successfully Seeded: {len(final_df)} rows from {final_df.index.min().date()}"
        except Exception as e:
            return f"Sync Failed: {str(e)}"
