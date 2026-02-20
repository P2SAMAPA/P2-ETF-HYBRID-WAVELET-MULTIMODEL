import io
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download
from datetime import datetime

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
HF_REPO_ID   = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_REPO_TYPE = "dataset"
HF_FILENAME  = "master_data.parquet"
DATA_START   = pd.Timestamp("2008-01-01")

ETF_TICKERS   = ["GLD", "SPY", "AGG", "TLT", "TBT", "VNQ", "SLV"]
STOOQ_ETF_MAP = {t: f"{t}.US" for t in ETF_TICKERS}

MACRO_CONFIG = {
    "VIX":        ("VIXCLS",        "^VIX"),
    "DXY":        ("DTWEXBGS",      "DXY"),
    "T10Y2Y":     ("T10Y2Y",        None),
    "TBILL_3M":   ("DTB3",          "^IRX"), 
    "IG_SPREAD": ("BAMLC0A0CM",    None),
    "HY_SPREAD": ("BAMLH0A0HYM2",  None),
}

# ---------------------------------------------------------------------------
# ETF FETCHERS
# ---------------------------------------------------------------------------
def _fetch_etf_stooq(tickers: list, start: pd.Timestamp) -> pd.DataFrame:
    try:
        frames = []
        for ticker in tickers:
            stooq_ticker = STOOQ_ETF_MAP[ticker]
            url = f"https://stooq.com/q/d/l/?s={stooq_ticker.lower()}&i=d"
            df  = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
            df.index = pd.DatetimeIndex(df.index)
            if "Close" not in df.columns: continue
            s = df["Close"].rename(ticker)
            frames.append(s[s.index >= start])
        if not frames: return pd.DataFrame()
        result = pd.concat(frames, axis=1).sort_index()
        return result.dropna(how="all")
    except Exception:
        return pd.DataFrame()

def _fetch_etf_yfinance(tickers: list, start: pd.Timestamp) -> pd.DataFrame:
    try:
        # Improved yfinance download to handle multi-tickers more reliably
        raw = yf.download(tickers, start=start, progress=False, group_by='ticker')
        frames = []
        for t in tickers:
            if t in raw.columns.levels[0]:
                frames.append(raw[t]['Close'].rename(t))
        if not frames: return pd.DataFrame()
        df = pd.concat(frames, axis=1)
        df.index = pd.DatetimeIndex(df.index)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# MACRO FETCHERS
# ---------------------------------------------------------------------------
def _fetch_macro_fred(fred: Fred, series_id: str, name: str, start: pd.Timestamp) -> pd.Series:
    try:
        s = fred.get_series(series_id, observation_start=start)
        s = pd.Series(s, name=name)
        s.index = pd.DatetimeIndex(s.index)
        return s
    except Exception:
        return pd.Series(dtype=float, name=name)

def _fetch_macro_stooq(ticker: str, name: str, start: pd.Timestamp) -> pd.Series:
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
        df  = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
        df.index = pd.DatetimeIndex(df.index)
        s = df["Close"].rename(name)
        return s[s.index >= start]
    except Exception:
        return pd.Series(dtype=float, name=name)

def _fetch_all_macros(fred: Fred, start: pd.Timestamp) -> pd.DataFrame:
    series_list = []
    for name, (fred_id, stooq_ticker) in MACRO_CONFIG.items():
        s = _fetch_macro_fred(fred, fred_id, name, start)
        if s.dropna().empty and stooq_ticker:
            s = _fetch_macro_stooq(stooq_ticker, name, start)
        series_list.append(s)
    macro_df = pd.concat(series_list, axis=1)
    macro_df.index = pd.DatetimeIndex(macro_df.index)
    return macro_df.ffill(limit=5)

# ---------------------------------------------------------------------------
# FEATURE LOADER CLASS
# ---------------------------------------------------------------------------
class FeatureLoader:
    def __init__(self, fred_key: str, hf_token: str = None):
        self.fred     = Fred(api_key=fred_key)
        self.hf_token = hf_token
        self.repo_id  = HF_REPO_ID

    def load_master(self) -> pd.DataFrame:
        try:
            path = hf_hub_download(
                repo_id   = self.repo_id,
                filename  = HF_FILENAME,
                repo_type = HF_REPO_TYPE,
                token     = self.hf_token # Added token to download if private
            )
            df = pd.read_parquet(path)
            df.index = pd.DatetimeIndex(df.index)
            return df
        except Exception:
            return pd.DataFrame()

    def sync_data(self) -> str:
        today = pd.Timestamp.now().normalize()
        master_df = self.load_master()
        
        is_full_seed = master_df.empty or "SOFR" in master_df.columns

        if not is_full_seed:
            last_date = master_df.index.max()
            if last_date >= (today - pd.Timedelta(days=1)):
                return "Incremental Refresh: Already Up to Date"
            start_fetch = last_date - pd.Timedelta(days=1) # Slight overlap to ensure continuity
        else:
            start_fetch = DATA_START

        try:
            etf_df = _fetch_etf_stooq(ETF_TICKERS, start_fetch)
            if etf_df.empty:
                etf_df = _fetch_etf_yfinance(ETF_TICKERS, start_fetch)

            if etf_df.empty: return "Sync Failed: ETF Source unavailable"

            macro_df = _fetch_all_macros(self.fred, start_fetch)
            combined = pd.concat([etf_df, macro_df], axis=1).ffill(limit=5)
            
            # Filter for trading days
            combined = combined[combined.index.dayofweek < 5]

            final_df = combined if is_full_seed else pd.concat([master_df, combined])
            final_df = final_df[~final_df.index.duplicated(keep="last")].sort_index()
            final_df.columns = [str(c).strip() for c in final_df.columns]

            # Upload to Hugging Face
            buf = io.BytesIO()
            final_df.to_parquet(buf)
            buf.seek(0)
            HfApi().upload_file(
                path_or_fileobj = buf,
                path_in_repo    = HF_FILENAME,
                repo_id         = self.repo_id,
                repo_type       = HF_REPO_TYPE,
                token           = self.hf_token,
            )
            return f"Success: {'Full Seed' if is_full_seed else 'Incremental'} synced. Rows: {len(final_df)}"
        except Exception as e:
            return f"Sync Failed: {str(e)}"

# ---------------------------------------------------------------------------
# APP WRAPPER (Modified to handle User-Triggered Force Refresh)
# ---------------------------------------------------------------------------
def load_raw_data(force_sync: bool = False):
    try:
        f_key = st.secrets["FRED_API_KEY"]
        h_token = st.secrets["HF_TOKEN"]
    except Exception:
        f_key = "PASTE_YOUR_KEY_HERE"
        h_token = None

    loader = FeatureLoader(fred_key=f_key, hf_token=h_token)
    
    # 1. If user clicked button OR SOFR exists, trigger the API fetcher
    if force_sync and h_token:
        st.info("🔄 Initiating External API Sync (FRED/Stooq/YF)...")
        loader.sync_data()
    
    # 2. Load the data (now potentially updated)
    df = loader.load_master()
    
    # 3. Fallback to live yfinance if HF is still empty
    if df.empty:
        df = _fetch_etf_yfinance(ETF_TICKERS, DATA_START)

    # 4. Process Returns
    for t in ETF_TICKERS:
        if t in df.columns:
            df[f"{t}_Ret"] = df[t].pct_change()

    # Drop only rows where returns are missing (usually just the first row)
    return df.dropna(subset=[f"{t}_Ret" for t in ETF_TICKERS if t in df.columns])
