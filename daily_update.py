#!/usr/bin/env python3
"""
Daily data update script.
Run this via GitHub Actions or manually.

What gets updated each run:
  - ETF prices          — last 10 days (buffer for late FRED updates)
  - Macro / rates       — last 10 days
  - ADV / dollar volume — recomputed from updated price + volume data
  - Options signals     — fetched LIVE for today (this is the primary way
                          options columns accumulate history day by day)
"""
import os
import sys
import yaml

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
    config  = load_config()
    seeding = config['seeding']
    daily   = config['daily_update']

    # Always use seeding symbols — daily.symbols is intentionally left empty
    symbols = seeding['symbols']

    print(f"Config loaded. Found {len(symbols)} symbols in seeding config.")
    print(f"Daily update config symbols: {daily.get('symbols', 'Not set (using seeding)')}")

    fred_key = os.environ.get('FRED_API_KEY')
    if not fred_key:
        print("Error: FRED_API_KEY not set in environment")
        sys.exit(1)

    hf_token = os.environ.get('HF_TOKEN')

    loader = FeatureLoader(fred_key=fred_key, hf_token=hf_token, symbols=symbols)
    print(f"Updating data for {len(symbols)} symbols: {symbols}")

    result = loader.sync_data(force=False)
    print(result)

    # Column summary after update (so you can verify options cols are accumulating)
    try:
        import pandas as pd
        df = loader.load_master()
        if not df.empty:
            adv_cols = [c for c in df.columns if "_ADV_" in c or "_DVOL_" in c]
            opt_cols = [c for c in df.columns if any(s in c for s in [
                "_IV_", "_SKEW_", "_PCR_", "_IMPLIED_", "_VRP_", "_GEX"])]
            # Check how many days of options data we have so far
            if opt_cols:
                opt_coverage = df[opt_cols].notna().any(axis=1).sum()
                print(f"  ADV columns       : {len(adv_cols)} "
                      f"(full history)")
                print(f"  Options columns   : {len(opt_cols)} "
                      f"({opt_coverage} days of live data so far)")
                print(f"  Latest date       : {df.index.max().date()}")
    except Exception:
        pass

    if daily.get('retrain_models', False):
        print("Retraining models...")
        os.system('python train_models.py')


if __name__ == "__main__":
    main()
