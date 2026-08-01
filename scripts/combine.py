#!/usr/bin/env python3
"""
combine.py — Drop Noise Columns & Combine All Datasets
========================================================
Reads every data/processed/<region>/<year>/<month>/data.csv,
drops identifier / leakage columns that would confuse the model,
adds a 'region' one-hot encoding as useful contextual feature,
then concatenates everything into one tidy CSV saved at:
    data/combined/combined_dataset.csv

Usage
-----
    python scripts/combine.py            # combine all
    python scripts/combine.py --dry-run  # show what would be dropped/kept
    python scripts/combine.py --help

Output columns (37 features + 17 labels = 54 total)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
COMBINED_DIR   = PROJECT_ROOT / "data" / "combined"
OUTPUT_FILE    = COMBINED_DIR / "combined_dataset.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Columns to DROP — identifiers, coordinates, leakage, admin metadata
# These add no predictive signal and can cause the model to overfit to IDs.
# ─────────────────────────────────────────────────────────────────────────────
DROP_COLS = [
    "system:index",          # grid cell ID string — purely administrative
    "grid_id",               # another ID field
    "latitude",              # raw coordinates → use region encoding instead
    "longitude",
    "year_month",            # date string — leakage / not generalizable
    "data_year",             # we add month as numeric feature instead
    "region",                # will be one-hot encoded as region_* below
    # raw monthly values already captured in aggregated _mean/_max etc.
    # but we KEEP chirps_precipitation_mm / mean_temperature_c /
    # solar_radiation_mj_m2_day because they ARE prediction 15/16/17 targets
    # and also serve as current-month context features for other predictions.
]

# ─────────────────────────────────────────────────────────────────────────────
# Feature columns to KEEP (37 agronomic features)
# ─────────────────────────────────────────────────────────────────────────────
STATIC_FEATURES = [
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
    "distance_to_surface_water_m",
    "soil_cec_cmol_kg_0_30cm",
    "soil_clay_pct_0_30cm",
    "soil_sand_pct_0_30cm",
    "soil_silt_pct_0_30cm",
    "soil_soc_g_kg_0_30cm",
    "soil_ph_h2o_0_30cm",
    "surface_water_occurrence_pct",
    "surface_water_seasonality_months",
]

INFRASTRUCTURE_FEATURES = [
    "distance_to_road_km",
    "road_density_km_per_sqkm",
    "distance_to_railway_km",
    "railway_density_km_per_sqkm",
    "distance_to_river_km",
    "river_density_km_per_sqkm",
    "urban_fraction",
    "builtup_fraction",
    "cropland_fraction",
    "non_cropland_fraction",
    "permanent_water_fraction",
    "population_density",
    "valid_agriculture_mask",
]

DYNAMIC_FEATURES = [
    # Current month (raw)
    "chirps_precipitation_mm",
    "mean_temperature_c",
    "solar_radiation_mj_m2_day",
    # Precipitation stats
    "chirps_precipitation_mm_mean",
    "chirps_precipitation_mm_max",
    "chirps_precipitation_mm_min",
    "chirps_precipitation_mm_range",
    "chirps_precipitation_mm_cv",
    # Soil moisture stats
    "era5_soil_moisture_m3_m3_mean",
    "era5_soil_moisture_m3_m3_max",
    "era5_soil_moisture_m3_m3_min",
    "era5_soil_moisture_m3_m3_cv",
    # Temperature stats
    "mean_temperature_c_mean",
    "mean_temperature_c_max",
    "mean_temperature_c_min",
    "mean_temperature_c_range",
    # Vegetation
    "ndvi_median_mean",
    "ndvi_median_max",
    "ndvi_median_min",
    "ndvi_median_growing_season_mean",
    # Water index
    "ndwi_mcf_median_mean",
    "ndwi_mcf_median_max",
    # SAR backscatter
    "s1_vh_db_median_mean",
    "s1_vv_db_median_mean",
    # Solar radiation stats
    "solar_radiation_mj_m2_day_mean",
    "solar_radiation_mj_m2_day_max",
]

# Temporal context feature (kept as numeric)
TEMPORAL_FEATURES = ["data_month"]

# Regional crop planting percentage features
CROP_PCT_FEATURES = [
    "crop_area_pct_monsoon_rice",
    "crop_area_pct_dry_season_rice",
    "crop_area_pct_maize",
    "crop_area_pct_sugarcane",
    "crop_area_pct_cassava",
    "crop_area_pct_durian",
    "crop_area_pct_mangosteen",
    "crop_area_pct_longan",
    "crop_area_pct_mango",
    "crop_area_pct_chili",
    "crop_area_pct_tomato",
    "crop_area_pct_black_gram",
    "crop_area_pct_green_gram",
    "crop_area_pct_pigeon_pea",
    "crop_area_pct_groundnut",
    "crop_area_pct_sesame",
    "crop_area_pct_rubber",
]

ALL_FEATURES = STATIC_FEATURES + INFRASTRUCTURE_FEATURES + DYNAMIC_FEATURES + TEMPORAL_FEATURES + CROP_PCT_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# Label columns (17 predictions)
# ─────────────────────────────────────────────────────────────────────────────
CROP_SUITABILITY_COLS = [
    "crop_suitability_monsoon_rice",
    "crop_suitability_dry_season_rice",
    "crop_suitability_maize",
    "crop_suitability_sugarcane",
    "crop_suitability_cassava",
    "crop_suitability_durian",
    "crop_suitability_mangosteen",
    "crop_suitability_longan",
    "crop_suitability_mango",
    "crop_suitability_chili",
    "crop_suitability_tomato",
    "crop_suitability_black_gram",
    "crop_suitability_green_gram",
    "crop_suitability_pigeon_pea",
    "crop_suitability_groundnut",
    "crop_suitability_sesame",
    "crop_suitability_rubber",
]

LABEL_COLS = CROP_SUITABILITY_COLS + [
    "crop_health_score",                 # regression 0-1
    "crop_yield_t_ha",                   # regression tons/ha
    "irrigation_need",                   # classification 0/1/2
    "current_month_precipitation_mm",    # regression
    "current_month_mean_temperature_c",  # regression
    "current_month_solar_rad_mj_m2_day", # regression
    "flood_risk_level",                  # classification 0/1/2
    "drought_risk_score",                # regression 0-1
    "heat_stress_risk",                  # classification 0/1/2
    "optimal_planting_month",            # classification 1..12
    "nitrogen_requirement_level",        # classification 0/1/2
    "phosphorus_requirement_level",      # classification 0/1/2
    "soil_erosion_risk",                 # classification 0/1/2
    "market_integration_score",          # regression 0-1
    "post_harvest_loss_risk",            # regression 0-1
    "supply_chain_efficiency",           # regression 0-1
    "cold_chain_potential",              # regression 0-1
    "agricultural_land_conversion_risk", # regression 0-1
    "urban_encroachment_risk",           # regression 0-1
    "irrigation_potential",              # regression 0-1
    "surface_water_occurrence",          # regression 0-1
    "water_scarcity_risk",               # regression 0-1
    "agricultural_gdp_forecast",         # regression 0-1
]

REGIONS = ["ayeyawaddy", "bago", "magway", "mandalay", "sagaing", "yangon"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def find_all_csvs():
    """Yield (path, region, year, month) for every processed CSV."""
    for region_dir in sorted(PROCESSED_ROOT.iterdir()):
        if not region_dir.is_dir():
            continue
        for year_dir in sorted(region_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                p = month_dir / "data.csv"
                if p.is_file():
                    yield p, region_dir.name, year_dir.name, month_dir.name


def clean_dataframe(df: pd.DataFrame, region: str, year: str, month: str) -> pd.DataFrame:
    """
    Drop noise columns, one-hot encode region, coerce numerics,
    and return a clean dataframe with feature + label columns only.
    """
    # ── 1. Add numeric month (temporal context)
    df["data_month"] = int(month)

    # ── 2. One-hot encode region (5 binary columns, drop first to avoid collinearity)
    for r in REGIONS:
        df[f"region_{r}"] = 1 if region == r else 0

    # ── 3. Select only the columns we need
    region_oh_cols = [f"region_{r}" for r in REGIONS]
    desired = ALL_FEATURES + region_oh_cols + LABEL_COLS
    present = [c for c in desired if c in df.columns]
    df = df[present].copy()

    # ── 4. Coerce all feature columns to numeric (fill missing with NaN)
    feat_cols = [c for c in ALL_FEATURES + region_oh_cols if c in df.columns]
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 5. Coerce numeric label columns
    numeric_labels = [
        "crop_health_score", "crop_yield_t_ha", "irrigation_need",
        "current_month_precipitation_mm", "current_month_mean_temperature_c",
        "current_month_solar_rad_mj_m2_day", "flood_risk_level",
        "drought_risk_score", "heat_stress_risk", "optimal_planting_month",
        "nitrogen_requirement_level", "phosphorus_requirement_level",
        "soil_erosion_risk", "market_integration_score", "post_harvest_loss_risk",
        "supply_chain_efficiency", "cold_chain_potential",
        "agricultural_land_conversion_risk", "urban_encroachment_risk",
        "irrigation_potential", "surface_water_occurrence",
        "water_scarcity_risk", "agricultural_gdp_forecast"
    ]
    for col in numeric_labels:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 6. Fill missing numeric features with column median (robust imputation)
    for col in feat_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # ── 7. Drop any rows where ALL label columns are NaN (unlabeled rows)
    label_present = [c for c in LABEL_COLS if c in df.columns]
    df = df.dropna(how="all", subset=label_present)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Combine all processed CSVs into one clean training dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print column plan without writing any files.")
    parser.add_argument("--no-region-encoding", action="store_true",
                        help="Skip one-hot region columns.")
    args = parser.parse_args()

    files = list(find_all_csvs())
    if not files:
        print("[ERROR] No processed CSVs found. Run preprocess.py + label.py first.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  Myanmar Agricultural ML — Combine & Clean Pipeline")
    print(f"{'='*65}")
    print(f"\n  Found {len(files)} CSV file(s) to combine\n")

    if args.dry_run:
        print("  COLUMNS TO DROP:")
        for c in DROP_COLS:
            print(f"    ✗  {c}")
        print(f"\n  FEATURE COLUMNS TO KEEP ({len(ALL_FEATURES)}):")
        for i, c in enumerate(ALL_FEATURES, 1):
            print(f"    {i:>2}. {c}")
        print(f"\n  REGION ENCODING COLUMNS (5 one-hot):")
        for r in REGIONS:
            print(f"       region_{r}")
        print(f"\n  LABEL COLUMNS ({len(LABEL_COLS)}):")
        for i, c in enumerate(LABEL_COLS, 1):
            print(f"    {i:>2}. {c}")
        print(f"\n  Total output columns: {len(ALL_FEATURES) + 5 + len(LABEL_COLS)}")
        return

    frames = []
    for path, region, year, month in files:
        print(f"  Loading {region:>12} / {year} / {month} ...", end=" ")
        try:
            raw = pd.read_csv(path, low_memory=False)
            if raw.empty:
                print("[SKIP - EMPTY]")
                continue
            cleaned = clean_dataframe(raw, region, year, month)
            frames.append(cleaned)
            print(f"{len(cleaned):>5} rows, {cleaned.shape[1]} cols")
        except Exception as e:
            print(f"[SKIP - ERROR: {e}]")

    combined = pd.concat(frames, ignore_index=True)

    # ── Global median imputation for any remaining NaNs across the combined set
    feat_cols = [c for c in combined.columns if c not in LABEL_COLS]
    for col in feat_cols:
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())

    # ── Save
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    num_regions = len(REGIONS)
    total_features = len(ALL_FEATURES) + num_regions
    print(f"\n{'─'*65}")
    print(f"  Combined dataset saved to: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  Total rows  : {len(combined):,}")
    print(f"  Total cols  : {combined.shape[1]}  ({total_features} features + {len(LABEL_COLS)} labels)")
    print(f"\n  Feature breakdown:")
    print(f"    Static terrain/soil  : {len(STATIC_FEATURES):>3}")
    print(f"    Infrastructure       : {len(INFRASTRUCTURE_FEATURES):>3}")
    print(f"    Dynamic climate/veg  : {len(DYNAMIC_FEATURES):>3}")
    print(f"    Temporal (month)     : {len(TEMPORAL_FEATURES):>3}")
    print(f"    Crop area shares     : {len(CROP_PCT_FEATURES):>3}")
    print(f"    Region one-hot       : {num_regions:>3}")
    print(f"    ─────────────────────────")
    print(f"    Total features       : {total_features:>3}")
    print(f"\n  Label breakdown:")
    print(f"    Crop suitability (cls): {len(CROP_SUITABILITY_COLS):>3}  (excellent/good/moderate/poor)")
    print(f"    Health score (reg)    :   1  (0.0 – 1.0)")
    print(f"    Yield t/ha   (reg)    :   1  (tons/ha)")
    print(f"    Irrigation   (cls)    :   1  (0/1/2)")
    print(f"    Monthly stats (reg)   :   3  (precip/temp/solar)")
    print(f"    Climate & Farm mgmt   :   7  (risk/planting/fertilizer/erosion)")
    print(f"    Market & Supply chain :   4  (integration/loss/efficiency/cold_chain)")
    print(f"    Urban & Land use      :   2  (conversion/encroachment)")
    print(f"    Water & River access  :   3  (potential/occurrence/scarcity)")
    print(f"    Agri GDP Forecast     :   1  (forecast score)")
    print(f"    ─────────────────────────")
    print(f"    Total labels         : {len(LABEL_COLS):>3}")

    # ── Missing value report
    miss = combined.isnull().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        print(f"\n  Remaining NaNs (label columns only — expected for pass-throughs):")
        for col, cnt in miss.items():
            pct = cnt / len(combined) * 100
            print(f"    {col:<50} {cnt:>5} ({pct:.1f}%)")

    print(f"\n  Next step: python scripts/train.py")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
