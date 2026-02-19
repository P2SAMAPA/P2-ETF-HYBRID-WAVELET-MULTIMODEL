import io
import pandas as pd
import numpy as np
import yfinance as yf
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
STOOQ_ETF_MAP = {t: f"{t}.US" for t in ETF_TICKERS}  # e.g. GLD -> GLD.US

# Macro config: { column_name: (fred_series_id, stooq_fallback_or_None) }
MACRO_CONFIG = {
    "VIX":       ("VIXCLS",        "^VIX"),   # CBOE Volatility Index
    "DXY":       ("DTWEXBGS",      "DXY"),    # Broad Dollar Index
    "T10Y2Y":    ("T10Y2Y",        None),     # 10Y-2Y Treasury spread
    "SOFR":      ("SOFR",          "^IRX"),   # SOFR / 13-week T-bill proxy
    "IG_SPREAD": ("BAMLC0A0CM",    None),     # IG Corp OAS spread
    "HY_SPREAD": ("BAMLH0A0HYM2",  None),     # HY OAS spread
}


# ---------------------------------------------------------------------------
# ETF FETCHERS
# ---------------------------------------------------------------------------

def _fetch_etf_stooq(tickers: list, start: pd.Timestamp) -> pd.DataFrame:
    """Fetch ETF closing prices from Stooq. Returns empty DF on failure."""
    try:
        frames = []
        for ticker in tickers:
            stooq_ticker = STOOQ_ETF_MAP[ticker]
            url = f"https://stooq.com/q/d/l/?s={stooq_ticker.lower()}&i=d"
            df  = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
            df.index = pd.DatetimeIndex(df.index)
            if "Close" not in df.columns:
                print(f"Stooq: no Close column for {stooq_ticker} — aborting Stooq fetch")
                return pd.DataFrame()
            s = df["Close"].rename(ticker)
            frames.append(s[s.index >= start])

        result = pd.concat(frames, axis=1).sort_index()
        result.dropna(how="all", inplace=True)
        print(f"Stooq ETF OK: {len(result)} rows")
        return result
    except Exception as e:
        print(f"Stooq ETF fetch failed: {e}")
        return pd.DataFrame()


def _fetch_etf_yfinance(tickers: list, start: pd.Timestamp) -> pd.DataFrame:
    """yfinance fallback for ETF closing prices."""
    try:
        raw = yf.download(tickers, start=start, auto_adjust=False, progress=False)
        closes = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            closes.columns = tickers
        closes.index = pd.DatetimeIndex(closes.index)
        closes = closes[closes.index >= start]
        closes.dropna(how="all", inplace=True)
        print(f"yfinance ETF OK: {len(closes)} rows")
        return closes
    except Exception as e:
        print(f"yfinance ETF fetch failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# MACRO FETCHERS
# ---------------------------------------------------------------------------

def _fetch_macro_fred(fred: Fred, series_id: str, name: str,
                      start: pd.Timestamp) -> pd.Series:
    """Fetch a single macro series from FRED."""
    try:
        s = fred.get_series(series_id, observation_start=start)
        s = pd.Series(s, name=name)
        s.index = pd.DatetimeIndex(s.index)
        print(f"FRED OK: {name} ({series_id}) — {len(s)} obs")
        return s
    except Exception as e:
        print(f"FRED failed for {name} ({series_id}): {e}")
        return pd.Series(dtype=float, name=name)


def _fetch_macro_stooq(ticker: str, name: str, start: pd.Timestamp) -> pd.Series:
    """Stooq fallback for a single macro series."""
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
        df  = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
        df.index = pd.DatetimeIndex(df.index)
        s = df["Close"].rename(name)
        s = s[s.index >= start]
        print(f"Stooq macro fallback OK: {name} ({ticker}) — {len(s)} obs")
        return s
    except Exception as e:
        print(f"Stooq macro fallback failed for {name} ({ticker}): {e}")
        return pd.Series(dtype=float, name=name)


def _fetch_all_macros(fred: Fred, start: pd.Timestamp) -> pd.DataFrame:
    """
    Fetches all macro signals: FRED first, Stooq fallback where available.
    For series with no Stooq equivalent (T10Y2Y, IG_SPREAD, HY_SPREAD),
    forward-fills from last known value if FRED fails — never drops the row.
    """
    series_list = []

    for name, (fred_id, stooq_ticker) in MACRO_CONFIG.items():
        s = _fetch_macro_fred(fred, fred_id, name, start)

        if s.dropna().empty:
            if stooq_ticker:
                print(f"FRED empty for {name} — trying Stooq fallback ({stooq_ticker})...")
                s = _fetch_macro_stooq(stooq_ticker, name, start)
            else:
                print(f"FRED empty for {name} — no Stooq equivalent, will forward-fill.")

        series_list.append(s)

    macro_df = pd.concat(series_list, axis=1)
    macro_df.index = pd.DatetimeIndex(macro_df.index)
    # Forward-fill with limit=5 to handle weekends/FRED publication lags
    # without propagating stale values dangerously far
    macro_df = macro_df.ffill(limit=5)
    return macro_df


# ---------------------------------------------------------------------------
# FEATURE LOADER
# ---------------------------------------------------------------------------

class FeatureLoader:
    def __init__(self, fred_key: str, hf_token: str):
        """
        Parameters
        ----------
        fred_key : from st.secrets["FRED_API_KEY"]
        hf_token : from st.secrets["HF_TOKEN"] — only needed for uploads,
                   reads are public.
        """
        self.fred     = Fred(api_key=fred_key)
        self.hf_token = hf_token
        self.repo_id  = HF_REPO_ID

    def load_master(self) -> pd.DataFrame:
        """
        Reads master_data.parquet from the public HuggingFace dataset.
        No token required for public datasets.
        Returns empty DataFrame if not found.
        """
        try:
            path = hf_hub_download(
                repo_id   = self.repo_id,
                filename  = HF_FILENAME,
                repo_type = HF_REPO_TYPE,
            )
            df = pd.read_parquet(path)
            df.index = pd.DatetimeIndex(df.index)
            print(f"Master parquet loaded: {len(df)} rows, cols: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Could not load master parquet (will full-seed): {e}")
            return pd.DataFrame()

    def sync_data(self) -> str:
        """
        Syncs market data to HuggingFace:
          - ETF prices   : Stooq first → yfinance fallback
          - Macro signals: FRED first  → Stooq fallback (ffill if no Stooq)
          - Stores RAW PRICES (returns computed downstream in processor.py)
          - Incremental refresh: fetches only new rows since last known date
          - Full seed: fetches from 2008-01-01

        Returns a status string.
        """
        today = pd.Timestamp.now().normalize()

        # --- Load existing master ---
        master_df    = self.load_master()
        is_full_seed = master_df.empty or len(master_df) < 1000

        if not is_full_seed:
            last_date = master_df.index.max()
            if last_date >= (today - pd.Timedelta(days=1)):
                return "Incremental Refresh: Already Up to Date"
            # Start from day after last row — no overlap, no boundary return issue
            start_fetch = last_date + pd.Timedelta(days=1)
        else:
            start_fetch = DATA_START

        print(f"{'Full Seed' if is_full_seed else 'Incremental'} from {start_fetch.date()}")

        try:
            # --- ETF prices: Stooq first, yfinance fallback ---
            etf_df = _fetch_etf_stooq(ETF_TICKERS, start_fetch)
            if etf_df.empty or len(etf_df) < 2:
                print("Stooq insufficient — falling back to yfinance...")
                etf_df = _fetch_etf_yfinance(ETF_TICKERS, start_fetch)

            if etf_df.empty:
                return "Sync Failed: Could not fetch ETF data from Stooq or yfinance"

            missing_etfs = [t for t in ETF_TICKERS if t not in etf_df.columns]
            if missing_etfs:
                print(f"Warning: Missing ETF columns: {missing_etfs}")

            # --- Macro signals: FRED first, Stooq fallback ---
            macro_df = _fetch_all_macros(self.fred, start_fetch)
            if macro_df.empty:
                return "Sync Failed: Could not fetch any macro data"

            # --- Combine and clean ---
            combined = pd.concat([etf_df, macro_df], axis=1)
            combined = combined.ffill(limit=5)
            combined = combined[
                (combined.index.dayofweek < 5) &   # weekdays only
                (combined.index < today)            # no future dates
            ]

            # Drop rows where ALL ETF prices are missing
            etf_cols = [c for c in ETF_TICKERS if c in combined.columns]
            combined.dropna(subset=etf_cols, how="all", inplace=True)

            if combined.empty:
                return "Sync Failed: No valid rows after cleaning"

            # --- Merge with master ---
            final_df = combined if is_full_seed else pd.concat([master_df, combined])
            final_df = (
                final_df[~final_df.index.duplicated(keep="last")]
                .sort_index()
            )

            # --- Upload to HuggingFace ---
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

            label = "Full Seed" if is_full_seed else "Incremental Refresh"
            return f"Success: {label} complete. Rows: {len(final_df)}, Cols: {list(final_df.columns)}"

        except Exception as e:
            return f"Sync Failed: {str(e)}"
