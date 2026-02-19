import io
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TICKER MAPS
# ---------------------------------------------------------------------------

# Stooq appends ".US" for US-listed ETFs
ETF_STOOQ_MAP = {
    "GLD": "GLD.US",
    "SLV": "SLV.US",
    "SPY": "SPY.US",
    "AGG": "AGG.US",
    "TLT": "TLT.US",
    "TBT": "TBT.US",
    "VNQ": "VNQ.US",
}

# Macro signals: FRED series ID + optional Stooq fallback ticker.
# Credit spread series are FRED-only — no clean Stooq equivalent exists.
# On FRED failure those columns are forward-filled from last known master value.
MACRO_MAP = {
    "VIX":    {"fred": "VIXCLS",       "stooq": "^VIX"},
    "DXY":    {"fred": "DTWEXBGS",     "stooq": "DXY.FST"},
    "T10Y2Y": {"fred": "T10Y2Y",       "stooq": None},
    "SOFR":   {"fred": "SOFR",         "stooq": None},
    "IG_OAS": {"fred": "BAMLC0A0CM",   "stooq": None},    # IG Corp OAS spread
    "HY_OAS": {"fred": "BAMLH0A0HYM2", "stooq": None},    # High Yield OAS spread
}

FULL_SEED_START = pd.Timestamp("2008-01-01")


# ---------------------------------------------------------------------------
# LOW-LEVEL FETCHERS
# ---------------------------------------------------------------------------

def _fetch_stooq_etf(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch a single ETF close price series from Stooq."""
    stooq_ticker = ETF_STOOQ_MAP.get(ticker, f"{ticker}.US")
    url = (
        f"https://stooq.com/q/d/l/?s={stooq_ticker.lower()}"
        f"&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}&i=d"
    )
    try:
        df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
        if df.empty or "Close" not in df.columns:
            raise ValueError("Empty or malformed Stooq response")
        s = df["Close"].dropna()
        s.name = ticker
        log.info(f"Stooq OK: {ticker} — {len(s)} rows")
        return s
    except Exception as e:
        log.warning(f"Stooq FAILED for {ticker}: {e}")
        return pd.Series(name=ticker, dtype=float)


def _fetch_yfinance_etf(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fallback: fetch a single ETF raw close price series from yfinance."""
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if raw.empty or "Close" not in raw.columns:
            raise ValueError("Empty yfinance response")
        s = raw["Close"].dropna()
        s.name = ticker
        s.index = pd.to_datetime(s.index)
        log.info(f"yfinance OK: {ticker} — {len(s)} rows")
        return s
    except Exception as e:
        log.warning(f"yfinance FAILED for {ticker}: {e}")
        return pd.Series(name=ticker, dtype=float)


def _fetch_etf(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Stooq first → yfinance fallback for a single ETF."""
    s = _fetch_stooq_etf(ticker, start, end)
    if s.empty:
        log.info(f"Falling back to yfinance for {ticker}")
        s = _fetch_yfinance_etf(ticker, start, end)
    return s


def _fetch_fred_macro(name: str, fred_id: str, fred: Fred,
                      start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch a single macro series from FRED."""
    try:
        s = fred.get_series(fred_id, observation_start=start, observation_end=end)
        s = s.dropna()
        s.index = pd.to_datetime(s.index)
        s.name = name
        log.info(f"FRED OK: {name} ({fred_id}) — {len(s)} rows")
        return s
    except Exception as e:
        log.warning(f"FRED FAILED for {name} ({fred_id}): {e}")
        return pd.Series(name=name, dtype=float)


def _fetch_stooq_macro(name: str, stooq_ticker: str,
                       start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Stooq fallback for macro signals that have a Stooq equivalent."""
    url = (
        f"https://stooq.com/q/d/l/?s={stooq_ticker.lower()}"
        f"&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}&i=d"
    )
    try:
        df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
        if df.empty or "Close" not in df.columns:
            raise ValueError("Empty or malformed Stooq response")
        s = df["Close"].dropna()
        s.name = name
        log.info(f"Stooq macro OK: {name} — {len(s)} rows")
        return s
    except Exception as e:
        log.warning(f"Stooq macro FAILED for {name}: {e}")
        return pd.Series(name=name, dtype=float)


def _fetch_macro(name: str, fred: Fred,
                 start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    FRED first → Stooq fallback for a single macro signal.
    If no Stooq equivalent exists (stooq=None), returns empty Series on
    FRED failure — caller handles forward-fill from master history.
    """
    cfg = MACRO_MAP[name]
    s = _fetch_fred_macro(name, cfg["fred"], fred, start, end)
    if s.empty and cfg["stooq"] is not None:
        log.info(f"Falling back to Stooq for macro: {name}")
        s = _fetch_stooq_macro(name, cfg["stooq"], start, end)
    if s.empty:
        log.warning(
            f"{name}: both FRED and Stooq failed — "
            f"will forward-fill from last known master value."
        )
    return s


# ---------------------------------------------------------------------------
# MAIN LOADER CLASS
# ---------------------------------------------------------------------------

class FeatureLoader:
    def __init__(self, fred_key: str, hf_token: str, repo_id: str):
        self.fred        = Fred(api_key=fred_key)
        self.hf_token    = hf_token
        self.repo_id     = repo_id
        self.etf_tickers = list(ETF_STOOQ_MAP.keys())   # GLD SLV SPY AGG TLT TBT VNQ
        self.macro_names = list(MACRO_MAP.keys())        # VIX DXY T10Y2Y SOFR IG_OAS HY_OAS

    # ------------------------------------------------------------------
    # HUGGINGFACE I/O
    # ------------------------------------------------------------------

    def _load_master(self) -> tuple:
        """
        Loads master_data.parquet from HuggingFace.
        Returns (DataFrame, is_incremental: bool).
        Raises specific exceptions instead of bare except.
        """
        try:
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename="master_data.parquet",
                repo_type="dataset",
                token=self.hf_token
            )
            master_df = pd.read_parquet(path)
            master_df.index = pd.to_datetime(master_df.index)
            if len(master_df) > 1000:
                log.info(f"Loaded master dataset: {len(master_df)} rows")
                return master_df, True
            log.warning("Master dataset has fewer than 1000 rows — treating as full seed")
            return master_df, False
        except Exception as e:
            log.warning(f"Could not load master from HuggingFace: {e}")
            return pd.DataFrame(), False

    def _upload_master(self, df: pd.DataFrame) -> bool:
        """Uploads master DataFrame to HuggingFace as parquet."""
        try:
            buf = io.BytesIO()
            df.to_parquet(buf)
            buf.seek(0)   # seek before passing to avoid materialising full bytes in memory
            HfApi().upload_file(
                path_or_fileobj=buf,
                path_in_repo="master_data.parquet",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.hf_token
            )
            log.info(f"Uploaded master dataset: {len(df)} rows")
            return True
        except Exception as e:
            log.error(f"Upload failed: {e}")
            return False

    # ------------------------------------------------------------------
    # FETCH ALL DATA FOR A DATE RANGE
    # ------------------------------------------------------------------

    def _fetch_all(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetches all ETF prices (Stooq→yfinance) and macro signals (FRED→Stooq)
        for the given date range. Returns raw prices and macro levels — NOT returns.
        Returns are computed downstream in processor.py.
        """
        # --- ETFs: Stooq first → yfinance fallback ---
        etf_series = [_fetch_etf(t, start, end) for t in self.etf_tickers]
        etf_df = pd.concat(etf_series, axis=1)
        etf_df.index = pd.to_datetime(etf_df.index)

        # --- Macros: FRED first → Stooq fallback ---
        macro_series = [_fetch_macro(n, self.fred, start, end) for n in self.macro_names]
        macro_df = pd.concat(macro_series, axis=1)
        macro_df.index = pd.to_datetime(macro_df.index)

        # --- Combine ---
        combined = pd.concat([etf_df, macro_df], axis=1)
        combined = combined[combined.index < end].sort_index()

        # Forward-fill macros only (limit=5 days: covers weekends + holidays)
        # ETF prices intentionally NOT forward-filled — gaps remain visible
        combined[self.macro_names] = combined[self.macro_names].ffill(limit=5)

        # Drop rows where ALL ETF prices are missing (non-trading days)
        combined = combined.dropna(subset=self.etf_tickers, how='all')

        return combined

    # ------------------------------------------------------------------
    # PUBLIC SYNC ENTRY POINT
    # ------------------------------------------------------------------

    def sync_data(self) -> str:
        """
        Main sync method:
          1. Load existing master parquet from HuggingFace
          2. Determine full seed vs incremental refresh
          3. Fetch new data (Stooq→yfinance for ETFs, FRED→Stooq for macros)
          4. Merge, deduplicate, forward-fill macro gaps at join boundary
          5. Validate core columns present
          6. Upload updated master to HuggingFace

        Stores RAW PRICES + MACRO LEVELS. Returns computed in processor.py.
        """
        today = pd.Timestamp.now().normalize()
        master_df, is_incremental = self._load_master()

        # Already up to date check
        if is_incremental:
            last_date = master_df.index.max()
            if last_date >= (today - pd.Timedelta(days=1)):
                return "Incremental Refresh: Already Up to Date"
            # FIXED: +1 day so we never re-fetch and silently overwrite the last known row
            start_fetch = last_date + pd.Timedelta(days=1)
        else:
            start_fetch = FULL_SEED_START

        log.info(f"Fetching: {start_fetch.date()} → {today.date()}")

        new_df = self._fetch_all(start_fetch, today)

        if new_df.empty:
            return "Sync Failed: No new data returned from any source"

        # Merge with master
        if is_incremental and not master_df.empty:
            combined = pd.concat([master_df, new_df])
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            # Final forward-fill pass to cover macro gaps at the join boundary
            combined[self.macro_names] = combined[self.macro_names].ffill(limit=5)
        else:
            combined = new_df.sort_index()

        # Validate core ETFs are present and not entirely empty
        core_etfs = ["GLD", "SPY"]
        missing = [t for t in core_etfs if t not in combined.columns
                   or combined[t].isna().all()]
        if missing:
            return f"Sync Failed: Core ETF data missing or all-NaN for {missing}"

        if not self._upload_master(combined):
            return "Sync Failed: Upload to HuggingFace failed"

        status = "Incremental Refresh" if is_incremental else "Full Seed"
        return (
            f"Success: {status} complete | "
            f"Rows: {len(combined)} | "
            f"Columns: {list(combined.columns)} | "
            f"Range: {combined.index.min().date()} → {combined.index.max().date()}"
        )
