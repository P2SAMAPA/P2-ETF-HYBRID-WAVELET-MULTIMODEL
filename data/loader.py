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
        self.tickers = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
        # 'SOFR' is the standard daily overnight rate from FRED
        self.macros = {"DXY": "DTWEXBGS", "VIX": "VIXCLS", "T10Y2Y": "T10Y2Y", "SOFR": "SOFR"}

    def sync_data(self):
        try:
            path = hf_hub_download(repo_id=self.repo_id, filename="master_data.parquet", repo_type="dataset", token=self.hf_token)
            master_df = pd.read_parquet(path)
            last_date = master_df.index.max()
        except:
            master_df = pd.DataFrame()
            last_date = pd.Timestamp("2008-01-01")

        # Check if already refreshed today
        if last_date.date() >= pd.Timestamp.now().date():
            return "Already Refreshed"

        try:
            # Incremental Pull
            df_recent = yf.download(self.tickers, start=last_date, multi_level_index=False)
            if df_recent.empty: return "Refresh Failed: No new market data"
            
            recent_closes = df_recent[[c for c in df_recent.columns if 'Close' in c]]
            recent_closes.columns = [c.replace('Close_', '').strip() for c in recent_closes.columns]
            
            # Proper SOFR pull
            macro_df = pd.DataFrame({n: self.fred.get_series(fid, last_date) for n, fid in self.macros.items()})
            
            combined = pd.concat([recent_closes, macro_df], axis=1).ffill().dropna()
            final_df = pd.concat([master_df, combined]).drop_duplicates().sort_index()
            
            # Save to HF
            buf = io.BytesIO(); final_df.to_parquet(buf)
            HfApi().upload_file(path_or_fileobj=buf.getvalue(), path_in_repo="master_data.parquet", 
                                repo_id=self.repo_id, repo_type="dataset", token=self.hf_token)
            return "Data Refreshed Successfully"
        except Exception as e:
            return f"Refresh Failed: {str(e)}"
