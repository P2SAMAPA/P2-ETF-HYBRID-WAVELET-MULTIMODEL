import io
import os
import yaml
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG LOADER
# ---------------------------------------------------------------------------
def _load_seeding_config():
    """Load seeding configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            return config['seeding']
    except Exception:
        # Fallback to hardcoded defaults if file missing - include ALL symbols
        return {
            'symbols': [
                "GLD", "SPY", "AGG", "TLT", "VCIT", "LQD", "HYG", "VNQ", "SLV",
                "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
                "XME", "GDX", "IWM"
            ],
            'start_date': "2008-01-01",
            'end_date': "2026-03-31",
            'sources': {'price': 'yfinance', 'macro': 'fred'}
        }

# Load once at module level
SEEDING_CONFIG = _load_seeding_config()
ETF_TICKERS = SEEDING_CONFIG['symbols']
DATA_START = pd.Timestamp(SEEDING_CONFIG['start_date'])

# Stooq mapping (unchanged, but we need to map all tickers)
STOOQ_ETF_MAP = {t: f"{t}.US" for t in ETF_TICKERS}

MACRO_CONFIG = {
    "VIX":        ("VIXCLS",       "^VIX"),
    "DXY":        ("DTWEXBGS",      "DXY"),
    "T10Y2Y":     ("T10Y2Y",        None),
    "TBILL_3M":   ("DTB3",         "^IRX"),
    "IG_SPREAD":  ("BAMLC0A0CM",    None),
    "HY_SPREAD":  ("BAMLH0A0HYM2",  None),
}

# ---------------------------------------------------------------------------
# FETCHERS (unchanged, but will use ETF_TICKERS from config)
# ---------------------------------------------------------------------------
def _fetch_etf_stooq(tickers: list, start: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        stooq_ticker = STOOQ_ETF_MAP.get(ticker)
        if not stooq_ticker:
            continue
        try:
            url = f"https://stooq.com/q/d/l/?s={stooq_ticker.lower()}&i=d"
            df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
            df.index = pd.DatetimeIndex(df.index)
            if "Close" not in df.columns:
                continue
            s = df["Close"].rename(ticker)
            frames.append(s[s.index >= start])
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()

def _fetch_etf_yfinance(tickers: list, start: pd.Timestamp) -> pd.DataFrame:
    try:
        raw = yf.download(tickers, start=start, progress=False, group_by='ticker')
        frames = []
        for t in tickers:
            if t in raw.columns.levels[0]:
                frames.append(raw[t]['Close'].rename(t))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=1)
        df.index = pd.DatetimeIndex(df.index)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()

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
        df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
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
# LOADER CLASS (modified to accept symbols)
# ---------------------------------------------------------------------------
class FeatureLoader:
    def __init__(self, fred_key: str, hf_token: str = None, symbols: list = None):
        self.fred = Fred(api_key=fred_key)
        self.hf_token = hf_token
        self.repo_id = "P2SAMAPA/fi-etf-macro-signal-master-data"  # keep as is
        self.symbols = symbols if symbols is not None else ETF_TICKERS

    def load_master(self) -> pd.DataFrame:
        try:
            path = hf_hub_download(repo_id=self.repo_id, filename="master_data.parquet",
                                   repo_type="dataset", token=self.hf_token)
            df = pd.read_parquet(path)
            df.index = pd.DatetimeIndex(df.index)
            return df
        except Exception:
            return pd.DataFrame()

    def sync_data(self, force: bool = False) -> str:
        today = pd.Timestamp.now().normalize()
        master_df = self.load_master()

        # Expected columns: symbols + macro columns
        expected_cols = set(self.symbols) | set(MACRO_CONFIG.keys())
        has_all_cols = master_df.empty or all(c in master_df.columns for c in expected_cols)

        if not force and not master_df.empty and has_all_cols:
            last_date = master_df.index.max()
            if last_date >= today:
                return "Sync Status: Already Up to Date"
            start_fetch = last_date - pd.Timedelta(days=10)  # buffer
        else:
            # Full rebuild triggered
            start_fetch = DATA_START
            force = True

        try:
            # Fetch ETF data using self.symbols
            etf_df = _fetch_etf_stooq(self.symbols, start_fetch)
            if etf_df.empty or force:
                etf_df = _fetch_etf_yfinance(self.symbols, start_fetch)
            if etf_df.empty:
                return "Sync Failed: ETF Source unavailable"

            # Fetch macro data
            macro_df = _fetch_all_macros(self.fred, start_fetch)
            combined = pd.concat([etf_df, macro_df], axis=1).ffill(limit=5)
            combined = combined[combined.index.dayofweek < 5]  # weekdays only

            if master_df.empty or force:
                final_df = combined
            else:
                final_df = pd.concat([master_df, combined])
                final_df = final_df.loc[~final_df.index.duplicated(keep='last')].sort_index()

            # Upload to HF
            buf = io.BytesIO()
            final_df.to_parquet(buf)
            buf.seek(0)
            HfApi().upload_file(
                path_or_fileobj=buf,
                path_in_repo="master_data.parquet",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.hf_token
            )
            return f"Success: Synced thru {final_df.index.max().strftime('%Y-%m-%d')}"
        except Exception as e:
            return f"Sync Failed: {str(e)}"

@st.cache_data(show_spinner=False)
def load_raw_data(force_sync: bool = False):
    try:
        f_key = st.secrets["FRED_API_KEY"]
        h_token = st.secrets["HF_TOKEN"]
    except Exception:
        f_key = "PASTE_YOUR_KEY_HERE"
        h_token = None

    # Use the config-loaded symbols
    loader = FeatureLoader(fred_key=f_key, hf_token=h_token, symbols=ETF_TICKERS)
    msg = "Loaded from Cache"

    if force_sync and h_token:
        msg = loader.sync_data(force=True)

    df = loader.load_master()
    if df.empty:
        df = _fetch_etf_yfinance(ETF_TICKERS, DATA_START)

    # Ensure return columns exist
    for t in ETF_TICKERS:
        if t in df.columns:
            df[f"{t}_Ret"] = df[t].pct_change()

    return df.dropna(subset=["SPY_Ret"]), msg
