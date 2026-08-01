#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

p = Path("data/processed/features_dataset.parquet")
df = pd.read_parquet(p)

indices = {
    "Yangon": 0,
    "Ayeyawaddy": 133056,
    "Bago": 286407,
    "Magway": 439758,
    "Mandalay": 617858,
    "Sagaing": 769427
}

for r, idx in indices.items():
    row = df.iloc[idx]
    reg_cols = [c for c in df.columns if c.startswith("region_") and row[c] == 1]
    precip = row.get("chirps_precipitation_mm_mean", row.get("chirps_precipitation_mm", 0))
    temp = row.get("mean_temperature_c_mean", row.get("mean_temperature_c", 0))
    print(f"Index {idx:<8} | Preset: {r:<12} | Region Col: {reg_cols} | Precip: {precip:.1f} | Temp: {temp:.1f}")
