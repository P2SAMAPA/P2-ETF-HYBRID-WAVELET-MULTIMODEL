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

# ETF tickers — standard (yfinance) and Stooq format auto-mapped
ETF_TICKERS   = ["GLD", "SPY", "AGG", "TLT", "TBT", "VNQ", "SLV"]
STOOQ_ETF_MAP = {t: f"{t}.US" for t in ETF_TICKERS}

# Macro config: { column_name: (fred_series_id, stooq_fallback_or_None) }
MACRO_CONFIG = {
    "VIX":       ("VIXCLS",        "^VIX"),
    "DXY":       ("DTWEXBGS",      "DXY"),
    "T10Y2Y":    ("T10Y2Y",        None),
    "SOFR":      ("SOFR",          "^IRX"),
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
            if "Close" not in df.columns:
                return pd.DataFrame()
            s = df["Close"].rename(ticker)
            frames.append(s[s.index >= start])
        result = pd.concat(frames, axis=1).sort_index()
        result.dropna(how="all", inplace=True)
        return result
    except Exception:
        return pd.DataFrame()

def _fetch_etf_yfinance(tickers: list, start: pd.Timestamp) -> pd.DataFrame:
    try:
        raw = yf.download(tickers, start=start, auto_adjust=False, progress=False)
        closes = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            closes.columns = tickers
        closes.index = pd.DatetimeIndex(closes.index)
        closes = closes[closes.index >= start]
        closes.dropna(how="all", inplace=True)
        return closes
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
            )
            df = pd.read_parquet(path)
            df.index = pd.DatetimeIndex(df.index)
            return df
        except Exception:
            return pd.DataFrame()

    def sync_data(self) -> str:
        today = pd.Timestamp.now().normalize()
        master_df = self.load_master()
        is_full_seed = master_df.empty or len(master_df) < 1000

        if not is_full_seed:
            last_date = master_df.index.max()
            if last_date >= (today - pd.Timedelta(days=1)):
                return "Incremental Refresh: Already Up to Date"
            start_fetch = last_date + pd.Timedelta(days=1)
        else:
            start_fetch = DATA_START

        try:
            etf_df = _fetch_etf_stooq(ETF_TICKERS, start_fetch)
            if etf_df.empty:
                etf_df = _fetch_etf_yfinance(ETF_TICKERS, start_fetch)

            if etf_df.empty:
                return "Sync Failed: Could not fetch ETF data"

            macro_df = _fetch_all_macros(self.fred, start_fetch)
            combined = pd.concat([etf_df, macro_df], axis=1)
            combined = combined.ffill(limit=5)
            combined = combined[(combined.index.dayofweek < 5) & (combined.index < today)]

            final_df = combined if is_full_seed else pd.concat([master_df, combined])
            final_df = final_df[~final_df.index.duplicated(keep="last")].sort_index()

            # Clean columns
            final_df.columns = [str(c).strip() for c in final_df.columns]

            # Upload
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
            return f"Success: {'Full Seed' if is_full_seed else 'Incremental'} complete. Rows: {len(final_df)}"
        except Exception as e:
            return f"Sync Failed: {str(e)}"

# ---------------------------------------------------------------------------
# APP WRAPPER (Corrected Indentation)
# ---------------------------------------------------------------------------
def load_raw_data():
    """Satisfies the app.py import by initializing the FeatureLoader"""
    try:
        f_key = st.secrets["FRED_API_KEY"]
    except Exception:
        f_key = "PASTE_YOUR_KEY_HERE"

    loader = FeatureLoader(fred_key=f_key)
    df = loader.load_master()

    if df.empty:
        df = _fetch_etf_yfinance(ETF_TICKERS, DATA_START)

    # Ensure Return columns exist for SVR
    for t in ETF_TICKERS:
        if t in df.columns and f"{t}_Ret" not in df.columns:
            df[f"{t}_Ret"] = df[t].pct_change()

    return df.dropna()
