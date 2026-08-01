#!/usr/bin/env python3
"""Inspect a processed Parquet file and print first few rows.
Usage:
    python scripts/inspect.py <region> <year> <month>
"""
import sys
import pandas as pd
from pathlib import Path

if len(sys.argv) != 4:
    print("Usage: python scripts/peek_data.py <region> <year> <month>")
    sys.exit(1)
region, year, month = sys.argv[1:4]
file_path = Path('data/processed') / region / year / month / 'data.parquet'
if not file_path.is_file():
    print(f"File not found: {file_path}")
    sys.exit(1)

df = pd.read_parquet(file_path)
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nShape:', df.shape)
