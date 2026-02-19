import os
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
import io

class FeatureLoader:
    def __init__(self, fred_key, hf_token, repo_id):
        self.fred = Fred(api_key=fred_key)
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.tickers = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
        self.macros = {
            "DXY": "DTWEXBGS",
            "VIX": "VIXCLS",
            "T10Y2Y": "T10Y2Y",
            "SOFR": "SOFR"
        }

    def fetch_stooq(self, ticker):
        """Fetches historical data from Stooq."""
        url = f"https://stooq.com/q/d/l/?s={ticker}.us&i=d"
        df = pd.read_csv(url, index_col='Date', parse_dates=True)
        return df[['Close']].rename(columns={'Close': ticker}).astype(np.float32)

    def sync_data(self):
        """Incremental Sync Logic."""
        # 1. Try to download existing master data from HF
        try:
            path = hf_hub_download(repo_id=self.repo_id, filename="master_data.parquet", repo_type="dataset", token=self.hf_token)
            master_df = pd.read_parquet(path)
            last_date = master_df.index.max()
        except:
            master_df = pd.DataFrame()
            last_date = pd.Timestamp("2008-01-01")

        # 2. Fetch missing ETF data
        new_etf_data = []
        for t in self.tickers:
            df = yf.download(t, start=last_date).astype(np.float32)
            new_etf_data.append(df['Close'].rename(t))
        
        etf_combined = pd.concat(new_etf_data, axis=1).dropna()

        # 3. Fetch missing Macro data
        new_macro_data = {}
        for name, fred_id in self.macros.items():
            s = self.fred.get_series(fred_id, observation_start=last_date)
            new_macro_data[name] = s.astype(np.float32)
        
        macro_df = pd.DataFrame(new_macro_data)
        
        # 4. Merge and Upload
        combined = pd.concat([etf_combined, macro_df], axis=1).ffill().dropna()
        final_df = pd.concat([master_df, combined]).drop_duplicates()
        
        # Memory-efficient save
        buffer = io.BytesIO()
        final_df.to_parquet(buffer)
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=buffer.getvalue(),
            path_in_repo="master_data.parquet",
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.hf_token
        )
        return "Sync Complete"
