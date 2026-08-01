#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

p = Path("data/processed/features_dataset.parquet")
df = pd.read_parquet(p)

regions = ["ayeyawaddy", "bago", "magway", "mandalay", "sagaing", "yangon"]

print("EXACT START ROW INDEX PER REGION:")
for r in regions:
    col = f"region_{r}"
    if col in df.columns:
        match_idx = df[df[col] == 1].index
        if len(match_idx) > 0:
            first_idx = match_idx[0]
            count = len(match_idx)
            row = df.iloc[first_idx]
            precip = row.get("chirps_precipitation_mm_mean", 0)
            temp = row.get("mean_temperature_c_mean", 0)
            print(f"Region: {r.capitalize():<12} | First Index: {first_idx:<8} | Count: {count:<8} | Precip: {precip:.1f} | Temp: {temp:.1f}")
