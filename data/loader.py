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
        # Standard tickers for Yahoo Finance
        self.tickers = ["TLT", "TBT", "VNQ", "SLV", "GLD", "SPY", "AGG"]
        # FRED IDs for Macro features
        self.macros = {
            "DXY": "DTWEXBGS", 
            "VIX": "VIXCLS", 
            "T10Y2Y": "T10Y2Y", 
            "SOFR": "SOFR"
        }

    def sync_data(self, force_full_reseed=False):
        """
        Syncs data from Stooq/Yahoo and FRED. 
        If force_full_reseed is True, it wipes the existing file and pulls from 2008.
        """
        try:
            # 1. Attempt to load existing master data
            path = hf_hub_download(
                repo_id=self.repo_id, 
                filename="master_data.parquet", 
                repo_type="dataset", 
                token=self.hf_token
            )
            master_df = pd.read_parquet(path)
            
            # If the file exists but is suspiciously small, trigger a reseed
            if len(master_df) < 100:
                force_full_reseed = True
            
            last_date = master_df.index.max()
        except Exception:
            master_df = pd.DataFrame()
            last_date = pd.Timestamp("2008-01-01")
            force_full_reseed = True

        # 2. Check if we actually need to update
        today = pd.Timestamp.now().normalize()
        if not force_full_reseed and last_date >= (today - pd.Timedelta(days=1)):
            return "Already Up to Date"

        try:
            start_fetch = pd.Timestamp("2008-01-01") if force_full_reseed else last_date
            
            # 3. Fetch Market Data (yfinance)
            # Use multi_level_index=False to simplify the dataframe structure
            raw_data = yf.download(self.tickers, start=start_fetch, multi_level_index=False)
            
            # FIX: Ensure columns are flat strings to avoid 'tuple' object errors
            data = raw_data[[c for c in raw_data.columns if 'Close' in c]].copy()
            
            # Clean column names (e.g., 'Close_GLD' -> 'GLD')
            new_cols = []
            for col in data.columns:
                col_str = str(col)
                clean_name = col_str.replace('Close_', '').replace('Close', '').strip()
                # If cleaning leaves it empty (can happen with single-ticker pulls), use the ticker list
                if not clean_name and len(self.tickers) == 1:
                    clean_name = self.tickers[0]
                new_cols.append(clean_name)
            data.columns = new_cols

            # 4. Fetch Macro Data (FRED)
            macro_frames = []
            for name, fred_id in self.macros.items():
                s = self.fred.get_series(fred_id, start_fetch)
                macro_frames.append(pd.Series(s, name=name))
            
            macro_df = pd.concat(macro_frames, axis=1).ffill()
            
            # 5. Merge and Cleanup
            combined = pd.concat([data, macro_df], axis=1).ffill().dropna()
            
            # Only keep completed days (strictly before today)
            combined = combined[combined.index < today]

            if force_full_reseed:
                final_df = combined
            else:
                final_df = pd.concat([master_df, combined])
            
            # Drop duplicates and sort
            final_df = final_df[~final_df.index.duplicated(keep='last')].sort_index()

            # 6. Upload to Hugging Face
            buf = io.BytesIO()
            final_df.to_parquet(buf)
            
            HfApi().upload_file(
                path_or_fileobj=buf.getvalue(),
                path_in_repo="master_data.parquet",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.hf_token
            )
            
            return f"Success: Seeded {len(final_df)} rows (Starting {final_df.index.min().date()})"

        except Exception as e:
            return f"Sync Failed: {str(e)}"
