#!/usr/bin/env python3
"""
scripts/export_inference_features.py — Exports Features-Only Dataset for Model Server
======================================================================================
Strips out the 40 target columns from data/combined/combined_dataset.csv,
creating a lightweight, high-speed feature dataset (features_dataset.parquet & features_dataset.csv)
specifically optimized for production inference.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMBINED_FILE = PROJECT_ROOT / "data" / "combined" / "combined_dataset.csv"
OUT_PARQUET   = PROJECT_ROOT / "data" / "processed" / "features_dataset.parquet"
OUT_CSV       = PROJECT_ROOT / "data" / "processed" / "features_dataset.csv"

ALL_TARGETS = [
    "crop_suitability_monsoon_rice", "crop_suitability_dry_season_rice", "crop_suitability_maize",
    "crop_suitability_sugarcane", "crop_suitability_cassava", "crop_suitability_durian",
    "crop_suitability_mangosteen", "crop_suitability_longan", "crop_suitability_mango",
    "crop_suitability_chili", "crop_suitability_tomato", "crop_suitability_black_gram",
    "crop_suitability_green_gram", "crop_suitability_pigeon_pea", "crop_suitability_groundnut",
    "crop_suitability_sesame", "crop_suitability_rubber", "crop_health_score", "crop_yield_t_ha",
    "irrigation_need", "current_month_precipitation_mm", "current_month_mean_temperature_c",
    "current_month_solar_rad_mj_m2_day", "flood_risk_level", "drought_risk_score",
    "heat_stress_risk", "optimal_planting_month", "nitrogen_requirement_level",
    "phosphorus_requirement_level", "soil_erosion_risk", "market_integration_score",
    "post_harvest_loss_risk", "supply_chain_efficiency", "cold_chain_potential",
    "agricultural_land_conversion_risk", "urban_encroachment_risk", "irrigation_potential",
    "surface_water_occurrence", "water_scarcity_risk", "agricultural_gdp_forecast"
]

def main():
    print("=====================================================================")
    print("      EXPORTING LIGHTWEIGHT FEATURES-ONLY INFERENCE DATASET          ")
    print("=====================================================================")
    print(f" Source File : {COMBINED_FILE}")

    if not COMBINED_FILE.exists():
        print(f"[ERROR] Source file not found: {COMBINED_FILE}")
        return

    print("Loading combined dataset...")
    df = pd.read_csv(COMBINED_FILE)
    print(f"Original Dataset Size: {len(df):,} rows x {len(df.columns)} columns.")

    # Identify feature columns (exclude targets)
    feature_cols = [c for c in df.columns if c not in ALL_TARGETS]
    print(f"Extracted {len(feature_cols)} input feature columns.")

    feat_df = df[feature_cols].copy()

    # Save to Parquet (High-Speed & Compressed)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    try:
        feat_df.to_parquet(OUT_PARQUET, index=False)
        print(f"  ✓ Saved Parquet dataset to: {OUT_PARQUET} ({OUT_PARQUET.stat().st_size / (1024*1024):.1f} MB)")
    except Exception as e:
        print(f"[WARN] Parquet export failed ({e}). Defaulting to CSV export.")

    # Save to CSV
    feat_df.to_csv(OUT_CSV, index=False)
    print(f"  ✓ Saved CSV dataset to    : {OUT_CSV} ({OUT_CSV.stat().st_size / (1024*1024):.1f} MB)")

    print("=====================================================================")
    print("  EXCELENT! Features-only dataset export complete!                  ")
    print("=====================================================================\n")

if __name__ == "__main__":
    main()
