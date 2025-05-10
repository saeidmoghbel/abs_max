#!/usr/bin/env python3
"""
Reports the number of missing values in the raw chromophore dataset:
- Missing SMILES ('Chromophore')
- Missing absorption maxima ('Absorption max (nm)')

This helps justify why rows with missing data are dropped in preprocessing.
"""
import os
import json
import logging

import pandas as pd
import config

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def main():
    setup_logging()
    # Load raw dataset
    raw_path = config.RAW_CSV
    if not os.path.isfile(raw_path):
        logging.error("Raw CSV not found at %s", raw_path)
        return
    df = pd.read_csv(raw_path)
    total = len(df)

    # Count missing
    missing_smiles = df['Chromophore'].isna().sum()
    missing_abs = df['Absorption max (nm)'].isna().sum()

    # Log results
    logging.info("Total raw entries: %d", total)
    logging.info("Missing SMILES: %d (%.1f%%)", missing_smiles, missing_smiles/total*100)
    logging.info("Missing Absorption max (nm): %d (%.1f%%)", missing_abs, missing_abs/total*100)

    # Save report to JSON
    report = {
        'total_entries': total,
        'missing_smiles': missing_smiles,
        'missing_absorption_nm': missing_abs,
        'pct_missing_smiles': round(missing_smiles/total*100, 2),
        'pct_missing_abs_nm': round(missing_abs/total*100, 2)
    }
    report_path = os.path.join(config.PROCESSED_DATA_DIR, 'missing_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logging.info("Saved missing data report to %s", report_path)

if __name__ == '__main__':
    main()
