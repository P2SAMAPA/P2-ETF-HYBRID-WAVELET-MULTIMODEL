#!/usr/bin/env python3
"""
Daily data update script.
Run this via GitHub Actions or manually.
"""

import os
import sys
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.loader import FeatureLoader

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    seeding = config['seeding']
    daily = config['daily_update']

    symbols = daily['symbols'] if daily['symbols'] else seeding['symbols']

    fred_key = os.environ.get('FRED_API_KEY')
    if not fred_key:
        print("Error: FRED_API_KEY not set in environment")
        sys.exit(1)

    hf_token = os.environ.get('HF_TOKEN')

    loader = FeatureLoader(fred_key=fred_key, hf_token=hf_token, symbols=symbols)

    print(f"Updating data for symbols: {symbols}")
    result = loader.sync_data(force=False)
    print(result)

    if daily.get('retrain_models', False):
        print("Retraining models...")
        os.system('python train_models.py')

if __name__ == "__main__":
    main()
