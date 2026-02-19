import io
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download

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
    "VIX":       ("VIXCLS",        "^VIX"),
    "DXY":       ("DTWEXBGS",      "DXY"),
    "T10Y2Y":    ("T10Y2Y",        None),
    "SOFR":      ("SOFR",          "^IRX"),
    "IG_SPREAD": ("BAMLC0A0CM",    None),
    "HY_SPREAD": ("BAMLH0A0HYM2",  None),
}

# ---------------------------------------------------------------------------
# ETF & MACRO FETCHERS (Helper functions remain as you had them)
# ---------------------------------------------------------------------------
def _fetch_etf_stooq(tickers, start):
    try:
        frames = []
        for ticker in tickers:
            stooq_ticker = STOOQ_ETF_MAP[ticker]
            url = f"https://stooq.com/q/d/l/?s={stooq_ticker.lower()}&i=d"
            df  = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
            df.index = pd.DatetimeIndex(df.index)
            if "Close" not in df.columns: return pd.DataFrame()
            s = df["Close"].rename(ticker)
            frames.append(s[s.index >= start])
        result = pd.concat(frames, axis=1).sort_index()
        return result
    except: return pd.DataFrame()

def _fetch_etf_yfinance(tickers, start):
    try:
        raw = yf.download(tickers, start=start, auto_adjust=False, progress=False)
        closes = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex): closes.columns = tickers
        closes.index = pd.DatetimeIndex(closes.index)
        return closes[closes.index >= start]
    except: return pd.DataFrame()

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
            path = hf_hub_download(repo_id=self.repo_id, filename=HF_FILENAME, repo_type=HF_REPO_TYPE)
            df = pd.read_parquet(path)
            df.index = pd.DatetimeIndex(df.index)
            return df
        except Exception as e:
            print(f"Loading failed: {e}")
            return pd.DataFrame()

# ---------------------------------------------------------------------------
# THE MISSING LINK: load_raw_data
# ---------------------------------------------------------------------------
def load_raw_data() -> pd.DataFrame:
    """
    This is the entry point app.py is looking for. 
    It initializes the FeatureLoader using Streamlit secrets and returns the data.
    """
    # 1. Get credentials from secrets
    try:
        f_key = st.secrets["FRED_API_KEY"]
    except:
        f_key = "PASTE_YOUR_FRED_KEY_HERE_IF_NOT_USING_SECRETS"

    # 2. Initialize and load
    loader = FeatureLoader(fred_key=f_key)
    df = loader.load_master()
    
    # 3. If the Parquet fails, fallback to a direct yfinance download
    if df.empty:
        df = _fetch_etf_yfinance(ETF_TICKERS, DATA_START)
        
    # 4. Ensure Return columns exist (The Engine expects Ticker_Ret)
    for t in ETF_TICKERS:
        if t in df.columns and f"{t}_Ret" not in df.columns:
            df[f"{t}_Ret"] = df[t].pct_change()
            
    return df.dropna()
    def load_raw_data():
    """Wrapper function to satisfy app.py import requirements"""
    # Use secrets for the API key if on Streamlit Cloud
    try:
        fred_key = st.secrets["FRED_API_KEY"]
    except:
        fred_key = "YOUR_KEY_HERE" # Fallback for local testing
        
    loader = FeatureLoader(fred_key=fred_key, hf_token=None)
    df = loader.load_master()
    
    # Critical: Ensure return columns exist for the SVR
    for t in ["GLD", "SPY", "AGG", "TLT", "TBT", "VNQ", "SLV"]:
        if t in df.columns:
            df[f"{t}_Ret"] = df[t].pct_change()
    return df.dropna()
