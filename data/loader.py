import os
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
        # We use .US for Stooq and standard tickers for yfinance
        self.tickers = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
        self.macros = {"DXY": "DTWEXBGS", "VIX": "VIXCLS", "T10Y2Y": "T10Y2Y", "SOFR": "SOFR"}

    def fetch_stooq_history(self, ticker):
        """Pure Stooq CSV download for historical accuracy."""
        url = f"https://stooq.com/q/d/l/?s={ticker}.us&i=d"
        try:
            df = pd.read_csv(url, index_col='Date', parse_dates=True)
            return df[['Close']].rename(columns={'Close': ticker}).astype(np.float32)
        except Exception:
            return pd.DataFrame()

    def sync_data(self):
        """Incremental Sync: Stooq for history, yfinance for 2026 updates."""
        try:
            # Download existing master data from HF
            path = hf_hub_download(repo_id=self.repo_id, filename="master_data.parquet", repo_type="dataset", token=self.hf_token)
            master_df = pd.read_parquet(path)
            last_date = master_df.index.max()
        except:
            # SEEDING PHASE: If no file exists, pull full history from Stooq
            stooq_dfs = [self.fetch_stooq_history(t) for t in self.tickers]
            master_df = pd.concat(stooq_dfs, axis=1).ffill().dropna()
            last_date = master_df.index.max()

        # REFRESH PHASE: Get 2026 daily data via yfinance
        # multi_level_index=False is the fix for the 'str' object error
        df_recent = yf.download(self.tickers, start=last_date, multi_level_index=False)
        
        # In recent yfinance, if multiple tickers are fetched, it returns 
        # a flat index but with Ticker_Attribute names. We filter for 'Close'.
        recent_closes = df_recent[[col for col in df_recent.columns if 'Close' in col]]
        # Clean column names from "Close_TLT" to "TLT"
        recent_closes.columns = [col.replace('Close_', '').replace('Close', '').strip() or self.tickers[0] for col in recent_closes.columns]

        # Fetch FRED Macros
        macro_data = {name: self.fred.get_series(fid, last_date).astype(np.float32) for name, fid in self.macros.items()}
        macro_df = pd.DataFrame(macro_data)

        # Merge & Upload
        combined = pd.concat([recent_closes, macro_df], axis=1).ffill().dropna()
        final_df = pd.concat([master_df, combined]).drop_duplicates().sort_index()

        buffer = io.BytesIO()
        final_df.to_parquet(buffer)
        
        HfApi().upload_file(
            path_or_fileobj=buffer.getvalue(),
            path_in_repo="master_data.parquet",
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.hf_token
        )
        return "Sync Complete (Stooq History + 2026 Refresh)"
