#!/usr/bin/env python3
"""
Initial seeding script.
Run this once to populate the HF dataset with full historical data.

What gets seeded:
  - ETF price history (2008-present) via yfinance / stooq
  - Macro data (FRED): VIX, DXY, T10Y2Y, TBILL_3M, IG_SPREAD, HY_SPREAD,
    DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS7, DGS10, DGS20, DGS30
  - ADV / dollar volume columns (rolling 20D, 63D, 252D) — computed from
    historical yfinance volume, so full history is available immediately
  - Options-derived scalar signals (IV_ATM, SKEW, PCR, implied move, VRP, GEX)
    NOTE: options signals are LIVE ONLY — yfinance does not provide historical
    options chains. These columns will be NaN for all historical rows and will
    populate from today's date onwards on each daily_update.py run.
    This is expected behaviour, not a bug.
"""
import os
import sys
import yaml
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Support both flat layout (data_loader.py) and package layout (data/loader.py)
try:
    from data.loader import FeatureLoader
except ImportError:
    from data_loader import FeatureLoader


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Seed the dataset with full history.')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--no-options', action='store_true',
                        help='Skip options fetching during seeding (faster, options are live-only anyway)')
    args = parser.parse_args()

    config  = load_config()
    seeding = config['seeding']
    symbols = seeding['symbols']

    fred_key = os.environ.get('FRED_API_KEY')
    if not fred_key:
        print("Error: FRED_API_KEY not set in environment")
        sys.exit(1)

    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not set — will run but upload will fail.")

    loader = FeatureLoader(fred_key=fred_key, hf_token=hf_token, symbols=symbols)

    print(f"\nSeeding data for {len(symbols)} symbols: {symbols}")
    print(f"Start date : {seeding['start_date']}")
    print(f"\nData layers that will be seeded:")
    print(f"  ✓ ETF prices          — full history from {seeding['start_date']}")
    print(f"  ✓ Macro / rates       — full history (FRED + stooq fallback)")
    print(f"  ✓ ADV / dollar volume — full history (computed from yfinance volume)")

    vol_cfg = seeding.get('volume', {})
    if vol_cfg.get('enabled', True):
        windows = vol_cfg.get('rolling_windows', [20, 63, 252])
        print(f"    Windows: {windows}D | Dollar volume: {vol_cfg.get('store_dollar_volume', True)}")

    opt_cfg = seeding.get('options', {})
    if opt_cfg.get('enabled', True) and not args.no_options:
        liquid = opt_cfg.get('liquid_options_symbols', [])
        print(f"  ⚠ Options signals     — LIVE ONLY (NaN for historical rows)")
        print(f"    Liquid symbols ({len(liquid)}): {liquid}")
        print(f"    Historical options data is not available via yfinance.")
        print(f"    Options columns will populate from today onwards via daily_update.py.")
    else:
        print(f"  — Options signals     — skipped (--no-options flag or disabled in config)")

    print(f"\nThis will REPLACE the existing HF dataset with full history.")

    if not args.yes:
        confirm = input("\nType 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return

    print("\nStarting seeding...\n")
    result = loader.sync_data(force=True)
    print(f"\nResult: {result}")

    # Column summary
    try:
        import pandas as pd
        df = loader.load_master()
        if not df.empty:
            price_cols  = [c for c in df.columns if c in symbols]
            macro_cols  = [c for c in df.columns if c in [
                "VIX","DXY","T10Y2Y","TBILL_3M","IG_SPREAD","HY_SPREAD",
                "DGS1MO","DGS3MO","DGS6MO","DGS1","DGS2","DGS5",
                "DGS7","DGS10","DGS20","DGS30"]]
            adv_cols    = [c for c in df.columns if "_ADV_" in c or "_DVOL_" in c]
            opt_cols    = [c for c in df.columns if any(s in c for s in [
                "_IV_","_SKEW_","_PCR_","_IMPLIED_","_VRP_","_GEX"])]
            other_cols  = [c for c in df.columns
                           if c not in price_cols + macro_cols + adv_cols + opt_cols]

            print(f"\n{'='*55}")
            print(f"  SEEDED DATASET SUMMARY")
            print(f"{'='*55}")
            print(f"  Rows (trading days) : {len(df):,}")
            print(f"  Date range          : {df.index.min().date()} → {df.index.max().date()}")
            print(f"  Total columns       : {len(df.columns)}")
            print(f"  ├─ Price columns    : {len(price_cols)}")
            print(f"  ├─ Macro columns    : {len(macro_cols)}")
            print(f"  ├─ ADV/volume cols  : {len(adv_cols)}")
            print(f"  ├─ Options cols     : {len(opt_cols)} "
                  f"({'live data from today' if len(opt_cols) > 0 else 'none yet — will populate on daily_update.py runs'})")
            print(f"  └─ Other columns    : {len(other_cols)}")
            if adv_cols:
                print(f"\n  ADV columns added: {adv_cols[:6]}{'...' if len(adv_cols) > 6 else ''}")
            if opt_cols:
                print(f"  Options columns  : {opt_cols[:6]}{'...' if len(opt_cols) > 6 else ''}")
            print(f"{'='*55}")
    except Exception as e:
        print(f"(Could not load summary: {e})")


if __name__ == "__main__":
    main()
