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
        self.tickers = ["TLT", "TBT", "VNQ", "SLV", "GLD", "SPY", "AGG"]
        self.macros = {"DXY": "DTWEXBGS", "VIX": "VIXCLS", "T10Y2Y": "T10Y2Y", "SOFR": "SOFR"}

    def sync_data(self):
        is_incremental = False
        try:
            path = hf_hub_download(repo_id=self.repo_id, filename="master_data.parquet", repo_type="dataset", token=self.hf_token)
            master_df = pd.read_parquet(path)
            # If we have more than 1000 rows, we consider it a healthy dataset for incremental refresh
            if len(master_df) > 1000:
                is_incremental = True
            last_date = master_df.index.max()
        except:
            master_df = pd.DataFrame()
            last_date = pd.Timestamp("2008-01-01")

        today = pd.Timestamp.now().normalize()
        if is_incremental and last_date >= (today - pd.Timedelta(days=1)):
            return "Incremental Refresh: Already Up to Date"

        try:
            start_fetch = last_date if is_incremental else pd.Timestamp("2008-01-01")
            
            # Fetching market data
            raw_data = yf.download(self.tickers, start=start_fetch, multi_level_index=False)
            data = raw_data[[c for c in raw_data.columns if 'Close' in c]].copy()
            data.columns = [str(c).replace('Close_', '').strip() for c in data.columns]
            
            # Fetching macros
            macro_frames = [pd.Series(self.fred.get_series(fid, start_fetch), name=n) for n, fid in self.macros.items()]
            macro_df = pd.concat(macro_frames, axis=1).ffill()
            
            combined = pd.concat([data, macro_df], axis=1).ffill().dropna()
            combined = combined[combined.index < today]

            final_df = combined if not is_incremental else pd.concat([master_df, combined])
            final_df = final_df[~final_df.index.duplicated(keep='last')].sort_index()

            # Upload
            buf = io.BytesIO()
            final_df.to_parquet(buf)
            HfApi().upload_file(path_or_fileobj=buf.getvalue(), path_in_repo="master_data.parquet", 
                                repo_id=self.repo_id, repo_type="dataset", token=self.hf_token)
            
            status_type = "Incremental Refresh" if is_incremental else "Full Seed"
            return f"Success: {status_type} done. Total rows: {len(final_df)}"

        except Exception as e:
            return f"Sync Failed: {str(e)}"
