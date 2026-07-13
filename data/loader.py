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
        return {
            'symbols': [
                "GLD", "SPY", "AGG", "TLT", "VCIT", "LQD", "HYG", "VNQ", "SLV", "IWF", "XSD", "XLB", "XBI",
                "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "SMH", "SOXX",
                "XME", "GDX", "IWM", "IWD", "IWO"
            ],
            'start_date': "2008-01-01",
            'end_date': "2035-03-31",
            'sources': {'price': 'yfinance', 'macro': 'fred'},
            'volume': {
                'enabled': True,
                'rolling_windows': [20, 63, 252],
                'store_dollar_volume': True,
            },
            'options': {
                'enabled': True,
                'liquid_options_symbols': [
                    "SPY", "QQQ", "IWM", "GLD", "TLT", "HYG",
                    "XLK", "XLE", "XLF", "GDX", "XBI", "SLV",
                    "VNQ", "LQD", "XLV", "XLU"
                ],
                'expiry_selection': {'target_dte': [7, 14, 30, 60, 90]},
                'moneyness': {'strike_range_pct': 0.20},
                'derived_signals': {
                    'iv_atm':           {'enabled': True, 'target_dte': [30, 60, 90]},
                    'iv_skew':          {'enabled': True, 'target_dte': [30]},
                    'iv_term_structure':{'enabled': True},
                    'iv_surface_level': {'enabled': True},
                    'put_call_ratio':   {'enabled': True, 'target_dte': [30]},
                    'put_call_vol_ratio':{'enabled': True, 'target_dte': [30]},
                    'implied_move':     {'enabled': True, 'target_dte': [7, 30]},
                    'vol_risk_premium': {'enabled': True},
                    'gamma_exposure':   {'enabled': True},
                },
            },
        }

SEEDING_CONFIG = _load_seeding_config()
ETF_TICKERS    = SEEDING_CONFIG['symbols']
DATA_START     = pd.Timestamp(SEEDING_CONFIG['start_date'])

VOL_CONFIG  = SEEDING_CONFIG.get('volume',  {})
OPT_CONFIG  = SEEDING_CONFIG.get('options', {})

ADV_WINDOWS          = VOL_CONFIG.get('rolling_windows', [20, 63, 252])
STORE_DOLLAR_VOL     = VOL_CONFIG.get('store_dollar_volume', True)
OPTIONS_ENABLED      = OPT_CONFIG.get('enabled', True)
LIQUID_OPT_SYMBOLS   = OPT_CONFIG.get('liquid_options_symbols', [
    "SPY", "QQQ", "IWM", "GLD", "TLT", "HYG",
    "XLK", "XLE", "XLF", "GDX", "XBI", "SLV",
    "VNQ", "LQD", "XLV", "XLU"
])
OPT_TARGET_DTE       = OPT_CONFIG.get('expiry_selection', {}).get('target_dte', [7, 14, 30, 60, 90])
OPT_STRIKE_RANGE     = OPT_CONFIG.get('moneyness', {}).get('strike_range_pct', 0.20)
OPT_SKIP_THRESHOLD   = OPT_CONFIG.get('skip_illiquid_threshold', 100)
DERIVED_SIG_CFG      = OPT_CONFIG.get('derived_signals', {})

STOOQ_ETF_MAP = {t: f"{t}.US" for t in ETF_TICKERS}

MACRO_CONFIG = {
    "VIX":       ("VIXCLS",       "^VIX"),
    "DXY":       ("DTWEXBGS",     "DXY"),
    "T10Y2Y":    ("T10Y2Y",       None),
    "TBILL_3M":  ("DTB3",         "^IRX"),
    "IG_SPREAD": ("BAMLC0A0CM",   None),
    "DGS1MO":    ("DGS1MO",       None),
    "DGS3MO":    ("DGS3MO",       None),
    "DGS6MO":    ("DGS6MO",       None),
    "DGS1":      ("DGS1",         None),
    "DGS2":      ("DGS2",         None),
    "DGS5":      ("DGS5",         None),
    "DGS7":      ("DGS7",         None),
    "DGS10":     ("DGS10",        None),
    "DGS20":     ("DGS20",        None),
    "DGS30":     ("DGS30",        None),
    "HY_SPREAD": ("BAMLH0A0HYM2", None),
}

# ---------------------------------------------------------------------------
# ETF PRICE FETCHERS (unchanged)
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


# ---------------------------------------------------------------------------
# VOLUME FETCHER
# New: fetches raw daily volume via yfinance and computes rolling ADV
# and dollar volume columns appended to the master DataFrame.
# ---------------------------------------------------------------------------
def _fetch_volume_yfinance(tickers: list, start: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch raw daily volume for all tickers via yfinance.
    Returns DataFrame of raw volumes indexed by date, one column per ticker
    named <TICKER>_RAW_VOL.
    """
    try:
        raw = yf.download(tickers, start=start, progress=False, group_by='ticker')
        frames = []
        for t in tickers:
            if t in raw.columns.levels[0] and 'Volume' in raw[t].columns:
                frames.append(raw[t]['Volume'].rename(f"{t}_RAW_VOL"))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=1)
        df.index = pd.DatetimeIndex(df.index)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def _compute_adv_columns(price_df: pd.DataFrame, volume_df: pd.DataFrame,
                          tickers: list, windows: list,
                          store_dollar_vol: bool) -> pd.DataFrame:
    """
    Compute rolling Average Daily Volume (ADV) and optionally Dollar Volume
    columns for each ticker and each rolling window.

    Columns produced:
        <TICKER>_ADV_<W>D      rolling mean of daily share volume over W days
        <TICKER>_DVOL_<W>D     rolling mean of daily dollar volume (price * vol) over W days
                               only if store_dollar_vol=True

    Returns DataFrame of ADV/DVOL columns aligned to price_df.index.
    """
    result = pd.DataFrame(index=price_df.index)
    for ticker in tickers:
        raw_vol_col = f"{ticker}_RAW_VOL"
        if raw_vol_col not in volume_df.columns:
            continue
        vol_series = volume_df[raw_vol_col].reindex(price_df.index).ffill()
        # Replace zeros with NaN (market closed / data gap)
        vol_series = vol_series.replace(0, np.nan)
        for w in windows:
            result[f"{ticker}_ADV_{w}D"] = vol_series.rolling(w, min_periods=max(1, w//2)).mean()
            if store_dollar_vol and ticker in price_df.columns:
                dollar_vol = vol_series * price_df[ticker]
                result[f"{ticker}_DVOL_{w}D"] = (
                    dollar_vol.rolling(w, min_periods=max(1, w//2)).mean()
                )
    return result


# ---------------------------------------------------------------------------
# OPTIONS FETCHER
# New: fetches options chain, computes scalar derived signals per symbol/date.
# Raw chain stored separately (options_data.parquet).
# Derived scalar signals returned as a single-row DataFrame for today.
# ---------------------------------------------------------------------------
def _nearest_expiry(expiries: list, target_dte: int,
                    today: pd.Timestamp, max_error: int = 5) -> str | None:
    """Find the expiry string closest to target_dte from today."""
    best, best_diff = None, float('inf')
    for exp in expiries:
        try:
            exp_dt = pd.Timestamp(exp)
            dte    = (exp_dt - today).days
            if dte < 0:
                continue
            diff = abs(dte - target_dte)
            if diff < best_diff and diff <= (target_dte + max_error):
                best_diff, best = diff, exp
        except Exception:
            continue
    return best


def _fetch_single_option_chain(ticker: str, today: pd.Timestamp,
                                target_dte_list: list,
                                strike_range_pct: float,
                                skip_threshold: int) -> dict:
    """
    Fetch options chain for one ticker, return dict of derived scalar signals.
    Keys match config column_template patterns.
    """
    signals = {}
    try:
        tk       = yf.Ticker(ticker)
        spot     = tk.fast_info.get('lastPrice') or tk.fast_info.get('regularMarketPrice')
        if spot is None or spot <= 0:
            return signals
        expiries = tk.options
        if not expiries:
            return signals

        chains_by_dte = {}
        for dte_target in target_dte_list:
            exp = _nearest_expiry(expiries, dte_target, today)
            if exp is None:
                continue
            try:
                chain = tk.option_chain(exp)
                calls = chain.calls.copy()
                puts  = chain.puts.copy()
                # Filter by moneyness
                lo, hi = spot * (1 - strike_range_pct), spot * (1 + strike_range_pct)
                calls = calls[(calls['strike'] >= lo) & (calls['strike'] <= hi)]
                puts  = puts[ (puts['strike']  >= lo) & (puts['strike']  <= hi)]
                # Skip if illiquid
                total_oi = calls['openInterest'].sum() + puts['openInterest'].sum()
                if total_oi < skip_threshold:
                    continue
                actual_dte = (pd.Timestamp(exp) - today).days
                chains_by_dte[dte_target] = {
                    'calls': calls, 'puts': puts,
                    'actual_dte': actual_dte, 'expiry': exp
                }
            except Exception:
                continue

        if not chains_by_dte:
            return signals

        # ── Derived signals ──────────────────────────────────────────────────

        cfg = DERIVED_SIG_CFG

        # IV ATM (30D, 60D, 90D)
        if cfg.get('iv_atm', {}).get('enabled', True):
            for dte in cfg.get('iv_atm', {}).get('target_dte', [30, 60, 90]):
                if dte not in chains_by_dte:
                    continue
                c = chains_by_dte[dte]['calls']
                atm_call = c.iloc[(c['strike'] - spot).abs().argsort()[:1]]
                if not atm_call.empty:
                    iv = atm_call['impliedVolatility'].values[0]
                    if not np.isnan(iv):
                        signals[f"{ticker}_IV_ATM_{dte}D"] = float(iv)

        # IV Skew — 25-delta risk reversal proxy (OTM call IV - OTM put IV)
        if cfg.get('iv_skew', {}).get('enabled', True):
            for dte in cfg.get('iv_skew', {}).get('target_dte', [30]):
                if dte not in chains_by_dte:
                    continue
                d = chains_by_dte[dte]
                c, p = d['calls'], d['puts']
                otm_call_strike = spot * 1.05
                otm_put_strike  = spot * 0.95
                otm_call = c.iloc[(c['strike'] - otm_call_strike).abs().argsort()[:1]]
                otm_put  = p.iloc[(p['strike']  - otm_put_strike).abs().argsort()[:1]]
                if not otm_call.empty and not otm_put.empty:
                    iv_call = otm_call['impliedVolatility'].values[0]
                    iv_put  = otm_put['impliedVolatility'].values[0]
                    if not np.isnan(iv_call) and not np.isnan(iv_put):
                        signals[f"{ticker}_SKEW_{dte}D"] = float(iv_call - iv_put)

        # IV Term Structure Slope (30D IV minus 90D IV)
        if cfg.get('iv_term_structure', {}).get('enabled', True):
            if 30 in chains_by_dte and 90 in chains_by_dte:
                iv30 = signals.get(f"{ticker}_IV_ATM_30D")
                iv90 = signals.get(f"{ticker}_IV_ATM_90D")
                if iv30 is not None and iv90 is not None:
                    signals[f"{ticker}_IV_TERM_SLOPE"] = float(iv30 - iv90)

        # IV Surface Level (mean IV across all available strikes in 30D chain)
        if cfg.get('iv_surface_level', {}).get('enabled', True):
            if 30 in chains_by_dte:
                d  = chains_by_dte[30]
                all_iv = pd.concat([
                    d['calls']['impliedVolatility'],
                    d['puts']['impliedVolatility']
                ]).dropna()
                if not all_iv.empty:
                    signals[f"{ticker}_IV_SURFACE_LEVEL"] = float(all_iv.mean())

        # Put-Call Ratio (Open Interest based)
        if cfg.get('put_call_ratio', {}).get('enabled', True):
            for dte in cfg.get('put_call_ratio', {}).get('target_dte', [30]):
                if dte not in chains_by_dte:
                    continue
                d = chains_by_dte[dte]
                call_oi = d['calls']['openInterest'].sum()
                put_oi  = d['puts']['openInterest'].sum()
                if call_oi > 0:
                    signals[f"{ticker}_PCR_OI_{dte}D"] = float(put_oi / call_oi)

        # Put-Call Ratio (Volume based)
        if cfg.get('put_call_vol_ratio', {}).get('enabled', True):
            for dte in cfg.get('put_call_vol_ratio', {}).get('target_dte', [30]):
                if dte not in chains_by_dte:
                    continue
                d = chains_by_dte[dte]
                call_vol = d['calls']['volume'].sum()
                put_vol  = d['puts']['volume'].sum()
                if call_vol > 0:
                    signals[f"{ticker}_PCR_VOL_{dte}D"] = float(put_vol / call_vol)

        # Implied Move (ATM straddle / spot)
        if cfg.get('implied_move', {}).get('enabled', True):
            for dte in cfg.get('implied_move', {}).get('target_dte', [7, 30]):
                if dte not in chains_by_dte:
                    continue
                d = chains_by_dte[dte]
                c, p = d['calls'], d['puts']
                atm_c = c.iloc[(c['strike'] - spot).abs().argsort()[:1]]
                atm_p = p.iloc[(p['strike']  - spot).abs().argsort()[:1]]
                if not atm_c.empty and not atm_p.empty:
                    straddle_price = (
                        atm_c['lastPrice'].values[0] + atm_p['lastPrice'].values[0]
                    )
                    if spot > 0:
                        signals[f"{ticker}_IMPLIED_MOVE_{dte}D"] = float(straddle_price / spot)

        # Gamma Exposure (GEX) proxy — dealer gamma in USD
        if cfg.get('gamma_exposure', {}).get('enabled', True):
            if 30 in chains_by_dte:
                d = chains_by_dte[30]
                c, p = d['calls'].copy(), d['puts'].copy()
                # GEX = sum(gamma * OI * 100 * spot) for calls, minus for puts
                # gamma estimated from Black-Scholes approx: not available directly
                # Use OI-weighted IV as GEX proxy when gamma not in chain
                call_gex = (c['openInterest'] * c['impliedVolatility']).sum()
                put_gex  = (p['openInterest'] * p['impliedVolatility']).sum()
                signals[f"{ticker}_GEX"] = float((call_gex - put_gex) * spot)

    except Exception:
        pass
    return signals


def _fetch_all_options_signals(tickers: list, today: pd.Timestamp) -> pd.Series:
    """
    Fetch options-derived scalar signals for all liquid tickers.
    Returns a pd.Series of scalar signal values, indexed by column name,
    representing today's snapshot.
    """
    all_signals = {}
    for ticker in tickers:
        sig = _fetch_single_option_chain(
            ticker, today,
            target_dte_list=OPT_TARGET_DTE,
            strike_range_pct=OPT_STRIKE_RANGE,
            skip_threshold=OPT_SKIP_THRESHOLD,
        )
        all_signals.update(sig)
    return pd.Series(all_signals, name=today)


def _compute_vol_risk_premium(master_df: pd.DataFrame, tickers: list,
                               realised_window: int = 21) -> pd.DataFrame:
    """
    Compute Volatility Risk Premium = IV_ATM_30D - annualised realised vol.
    Requires both IV_ATM_30D columns and price columns to be in master_df.
    Appended as <TICKER>_VRP_30D columns.
    """
    result = pd.DataFrame(index=master_df.index)
    if not cfg_enabled('vol_risk_premium'):
        return result
    for ticker in tickers:
        iv_col = f"{ticker}_IV_ATM_30D"
        if iv_col not in master_df.columns or ticker not in master_df.columns:
            continue
        log_ret    = np.log(master_df[ticker] / master_df[ticker].shift(1))
        realised   = log_ret.rolling(realised_window).std() * np.sqrt(252)
        result[f"{ticker}_VRP_30D"] = master_df[iv_col] - realised
    return result


def cfg_enabled(signal_key: str) -> bool:
    return DERIVED_SIG_CFG.get(signal_key, {}).get('enabled', True)


# ---------------------------------------------------------------------------
# MACRO FETCHERS (unchanged)
# ---------------------------------------------------------------------------
def _fetch_macro_fred(fred: Fred, series_id: str, name: str,
                       start: pd.Timestamp) -> pd.Series:
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
# LOADER CLASS
# ---------------------------------------------------------------------------
class FeatureLoader:
    def __init__(self, fred_key: str, hf_token: str = None, symbols: list = None):
        self.fred       = Fred(api_key=fred_key)
        self.hf_token   = hf_token
        self.repo_id    = "P2SAMAPA/fi-etf-macro-signal-master-data"
        self.symbols    = symbols if symbols is not None else ETF_TICKERS

    def load_master(self) -> pd.DataFrame:
        try:
            path = hf_hub_download(
                repo_id=self.repo_id, filename="master_data.parquet",
                repo_type="dataset", token=self.hf_token
            )
            df = pd.read_parquet(path)
            df.index = pd.DatetimeIndex(df.index)
            return df
        except Exception:
            return pd.DataFrame()

    def sync_data(self, force: bool = False) -> str:
        today       = pd.Timestamp.now().normalize()
        master_df   = self.load_master()

        expected_base = set(self.symbols) | set(MACRO_CONFIG.keys())
        has_all_cols  = master_df.empty or all(
            c in master_df.columns for c in expected_base
        )

        if not force and not master_df.empty and has_all_cols:
            last_date = master_df.index.max()
            if last_date >= today:
                return "Sync Status: Already Up to Date"
            start_fetch = last_date - pd.Timedelta(days=10)
        else:
            start_fetch = DATA_START
            force       = True

        try:
            # ── 1. Price data ─────────────────────────────────────────────
            etf_df = _fetch_etf_stooq(self.symbols, start_fetch)
            if etf_df.empty or force:
                etf_df = _fetch_etf_yfinance(self.symbols, start_fetch)
            if etf_df.empty:
                return "Sync Failed: ETF Source unavailable"

            # ── 2. Macro data ─────────────────────────────────────────────
            macro_df = _fetch_all_macros(self.fred, start_fetch)

            # ── 3. Volume data (ADV + dollar volume) ──────────────────────
            vol_enabled = VOL_CONFIG.get('enabled', True)
            adv_df = pd.DataFrame()
            if vol_enabled:
                raw_vol_df = _fetch_volume_yfinance(self.symbols, start_fetch)
                if not raw_vol_df.empty:
                    adv_df = _compute_adv_columns(
                        price_df=etf_df,
                        volume_df=raw_vol_df,
                        tickers=self.symbols,
                        windows=ADV_WINDOWS,
                        store_dollar_vol=STORE_DOLLAR_VOL,
                    )

            # ── 4. Options scalar signals ─────────────────────────────────
            # Only fetched for today (options chains are live/intraday).
            # For historical backfill, options signals are left NaN and
            # forward-filled from the first live run onwards.
            opt_df = pd.DataFrame()
            if OPTIONS_ENABLED:
                opt_signals = _fetch_all_options_signals(
                    LIQUID_OPT_SYMBOLS, today
                )
                if not opt_signals.empty:
                    # One row for today; will merge into master below
                    opt_df = pd.DataFrame([opt_signals], index=[today])

            # ── 5. Combine all sources ────────────────────────────────────
            frames = [etf_df, macro_df]
            if not adv_df.empty:
                frames.append(adv_df)
            combined = pd.concat(frames, axis=1).ffill(limit=5)
            combined = combined[combined.index.dayofweek < 5]

            # Merge options signals (today only)
            if not opt_df.empty:
                for col in opt_df.columns:
                    combined.loc[today, col] = opt_df.loc[today, col]

            # ── 6. Build final DataFrame ──────────────────────────────────
            if master_df.empty or force:
                final_df = combined
            else:
                # Add any new columns that appeared in combined but not master
                for col in combined.columns:
                    if col not in master_df.columns:
                        master_df[col] = np.nan
                final_df = pd.concat([master_df, combined])
                final_df = (final_df
                            .loc[~final_df.index.duplicated(keep='last')]
                            .sort_index())

            # ── 7. Compute VRP (needs both IV and price history) ──────────
            if OPTIONS_ENABLED and not final_df.empty:
                vrp_df = _compute_vol_risk_premium(final_df, LIQUID_OPT_SYMBOLS)
                if not vrp_df.empty:
                    # Use pd.concat instead of column-by-column insertion to
                    # avoid PerformanceWarning on highly fragmented DataFrames
                    final_df = pd.concat([final_df, vrp_df], axis=1)
                    # Drop duplicate columns (keep latest — VRP overwrites NaN placeholders)
                    final_df = final_df.loc[:, ~final_df.columns.duplicated(keep='last')]
                    # Defragment memory
                    final_df = final_df.copy()

            # ── 8. Upload to HuggingFace ──────────────────────────────────
            buf = io.BytesIO()
            final_df.to_parquet(buf)
            buf.seek(0)
            HfApi().upload_file(
                path_or_fileobj=buf,
                path_in_repo="master_data.parquet",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.hf_token,
            )

            os.makedirs('data', exist_ok=True)
            final_df.to_parquet('data/master_data.parquet')

            new_cols = [c for c in final_df.columns
                        if c not in (set(self.symbols) | set(MACRO_CONFIG.keys()))]
            return (
                f"Success: Synced through {final_df.index.max().strftime('%Y-%m-%d')} "
                f"| {len(final_df.columns)} total columns "
                f"| New signal columns: {len(new_cols)} "
                f"(ADV/DVOL + options scalars) — uploaded to HF dataset"
            )
        except Exception as e:
            return f"Sync Failed: {str(e)}"


# ---------------------------------------------------------------------------
# STREAMLIT CACHED LOADER
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data(force_sync: bool = False):
    try:
        f_key   = st.secrets["FRED_API_KEY"]
        h_token = st.secrets["HF_TOKEN"]
    except Exception:
        f_key   = "PASTE_YOUR_KEY_HERE"
        h_token = None

    loader = FeatureLoader(fred_key=f_key, hf_token=h_token, symbols=ETF_TICKERS)
    msg    = "Loaded from Cache"

    if force_sync and h_token:
        msg = loader.sync_data(force=True)

    df = loader.load_master()
    if df.empty:
        df = _fetch_etf_yfinance(ETF_TICKERS, DATA_START)

    for t in ETF_TICKERS:
        if t in df.columns:
            df[f"{t}_Ret"] = df[t].ffill().pct_change(fill_method=None)

    return df.dropna(subset=["SPY_Ret"]), msg
