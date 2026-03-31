#!/usr/bin/env python3
"""
Initial seeding script.
Run this once to populate the HF dataset with full historical data.
"""

import os
import sys
import yaml

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import FeatureLoader

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    seeding = config['seeding']
    symbols = seeding['symbols']

    fred_key = os.environ.get('FRED_API_KEY')
    if not fred_key:
        print("Error: FRED_API_KEY not set in environment")
        sys.exit(1)

    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not set. Will still run but may fail upload.")

    loader = FeatureLoader(fred_key=fred_key, hf_token=hf_token, symbols=symbols)

    print(f"Seeding data for symbols: {symbols}")
    print("This will replace the existing dataset with full history.")
    confirm = input("Type 'yes' to confirm: ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return

    result = loader.sync_data(force=True)
    print(result)

if __name__ == "__main__":
    main()
