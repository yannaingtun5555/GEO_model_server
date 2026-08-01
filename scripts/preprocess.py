#!/usr/bin/env python3
"""
preprocess.py — Preprocessing Pipeline
=======================================
Reads raw static + dynamic CSV files from data/raw/, merges them,
keeps only the feature columns defined in features.md, and saves
CSV files under data/processed/<region>/<year>/<month>/data.csv.

After running this script, run label.py to add the 17 prediction columns.

Usage
-----
    python scripts/preprocess.py                     # process all regions
    python scripts/preprocess.py --region ayeyawaddy  # single region
    python scripts/preprocess.py --help

Filename conventions (data/raw/)
---------------------------------
    <region>_static.csv                  one per region
    <region>_dynamic_<YYYY>_<MM>.csv     one per region per month
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
RAW_ROOT       = PROJECT_ROOT / "data" / "raw"
CSV6_ROOT      = PROJECT_ROOT / "data" / "CSV6" / "CSV6"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DATA_SEARCH_DIRS = [
    PROJECT_ROOT / "data" / "csv4",
    PROJECT_ROOT / "data" / "CSV4",
    PROJECT_ROOT / "data" / "CSV6" / "CSV6",
    PROJECT_ROOT / "data" / "CSV6",
    PROJECT_ROOT / "data" / "raw",
]

REGIONS = ["ayeyawaddy", "bago", "magway", "mandalay", "sagaing", "yangon"]

JOIN_KEY = "system:index"

# ---------------------------------------------------------------------------
# Exact feature columns to keep (from features.md)
# ---------------------------------------------------------------------------
# Static features (12 — latitude/longitude already in ID_COLS)
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
    # Infrastructure & Landcover features
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

# Dynamic raw columns (current-month values, passed through for predictions 15-17)
DYNAMIC_RAW_COLS = [
    "chirps_precipitation_mm",       # prediction 15 source
    "mean_temperature_c",            # prediction 16 source
    "solar_radiation_mj_m2_day",     # prediction 17 source
]

# Dynamic aggregated features (27) — computed within preprocess as rolling per-grid stats
# For now we keep the raw monthly values; label.py will use them for prediction 15-17.
# Aggregated features will be added in a future aggregation pass when multi-month data exists.
DYNAMIC_AGG_FEATURES = [
    # Precipitation
    "chirps_precipitation_mm_mean",
    "chirps_precipitation_mm_max",
    "chirps_precipitation_mm_min",
    "chirps_precipitation_mm_range",
    "chirps_precipitation_mm_cv",
    # Soil moisture
    "era5_soil_moisture_m3_m3_mean",
    "era5_soil_moisture_m3_m3_max",
    "era5_soil_moisture_m3_m3_min",
    "era5_soil_moisture_m3_m3_cv",
    # Temperature
    "mean_temperature_c_mean",
    "mean_temperature_c_max",
    "mean_temperature_c_min",
    "mean_temperature_c_range",
    # NDVI
    "ndvi_median_mean",
    "ndvi_median_max",
    "ndvi_median_min",
    "ndvi_median_growing_season_mean",
    # NDWI
    "ndwi_mcf_median_mean",
    "ndwi_mcf_median_max",
    # SAR backscatter
    "s1_vh_db_median_mean",
    "s1_vv_db_median_mean",
    # Solar radiation
    "solar_radiation_mj_m2_day_mean",
    "solar_radiation_mj_m2_day_max",
]

# Identifier / metadata columns kept for reference
ID_COLS = [JOIN_KEY, "grid_id", "latitude", "longitude", "year_month"]

# All columns to keep in processed output — deduplicated
_keep_seen: set = set()
_keep_ordered: list = []
for _c in ID_COLS + STATIC_FEATURES + DYNAMIC_RAW_COLS + DYNAMIC_AGG_FEATURES:
    if _c not in _keep_seen:
        _keep_ordered.append(_c)
        _keep_seen.add(_c)
KEEP_COLS = _keep_ordered


# ---------------------------------------------------------------------------
# Helper: compute single-month aggregated feature approximations
# ---------------------------------------------------------------------------
def compute_single_month_agg(df: pd.DataFrame, raw_col: str, prefix: str) -> pd.DataFrame:
    """
    When we only have one month of data, approximate the aggregated statistics
    using the available single monthly value. (Will be overridden in multi-month pass.)
    """
    if raw_col not in df.columns:
        for suffix in ["mean", "max", "min", "range", "cv"]:
            df[f"{prefix}_{suffix}"] = float("nan")
        return df
    v = pd.to_numeric(df[raw_col], errors="coerce")
    df[f"{prefix}_mean"]  = v
    df[f"{prefix}_max"]   = v
    df[f"{prefix}_min"]   = v
    df[f"{prefix}_range"] = 0.0
    df[f"{prefix}_cv"]    = 0.0   # single month → no variance
    return df


def compute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all aggregated features from raw monthly columns.
    For now assumes single-month slice (will be updated in multi-month pass).
    """
    # Precipitation aggregates
    df = compute_single_month_agg(df, "chirps_precipitation_mm",     "chirps_precipitation_mm")
    # Soil moisture
    df = compute_single_month_agg(df, "era5_soil_moisture_m3_m3",    "era5_soil_moisture_m3_m3")
    # Temperature
    df = compute_single_month_agg(df, "mean_temperature_c",          "mean_temperature_c")
    # NDVI
    df = compute_single_month_agg(df, "ndvi_median",                 "ndvi_median")
    if "ndvi_median_mean" in df.columns:
        df["ndvi_median_growing_season_mean"] = df["ndvi_median_mean"].where(
            df["ndvi_median_mean"].fillna(0) > 0.4, other=float("nan")
        ).fillna(df["ndvi_median_mean"])
    # NDWI
    df = compute_single_month_agg(df, "ndwi_mcf_median",             "ndwi_mcf_median")
    # Drop duplicated intermediate columns not in spec
    for col in ["ndvi_median_range", "ndvi_median_cv",
                "ndwi_mcf_median_min", "ndwi_mcf_median_range", "ndwi_mcf_median_cv"]:
        df.drop(columns=[col], errors="ignore", inplace=True)
    # SAR backscatter — single-month pass = mean
    for raw, agg in [("s1_vh_db_median", "s1_vh_db_median_mean"),
                     ("s1_vv_db_median", "s1_vv_db_median_mean")]:
        if raw in df.columns:
            df[agg] = pd.to_numeric(df[raw], errors="coerce")
        else:
            df[agg] = float("nan")
    # Solar radiation
    df = compute_single_month_agg(df, "solar_radiation_mj_m2_day",  "solar_radiation_mj_m2_day")
    for col in ["solar_radiation_mj_m2_day_min",
                "solar_radiation_mj_m2_day_range",
                "solar_radiation_mj_m2_day_cv"]:
        df.drop(columns=[col], errors="ignore", inplace=True)

    return df


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------
def compute_grid_time_series_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative or grouped feature statistics per grid_id over time
    when multi-month dataset is present in a single raw file.
    """
    if "grid_id" not in df.columns:
        return compute_aggregates(df)

    feature_map = [
        ("chirps_precipitation_mm", "chirps_precipitation_mm"),
        ("era5_soil_moisture_m3_m3", "era5_soil_moisture_m3_m3"),
        ("mean_temperature_c", "mean_temperature_c"),
        ("ndvi_median", "ndvi_median"),
        ("ndwi_mcf_median", "ndwi_mcf_median"),
        ("s1_vh_db_median", "s1_vh_db_median_mean"),
        ("s1_vv_db_median", "s1_vv_db_median_mean"),
        ("solar_radiation_mj_m2_day", "solar_radiation_mj_m2_day"),
    ]

    new_cols = {}
    for raw_col, prefix in feature_map:
        if raw_col not in df.columns:
            continue

        v = pd.to_numeric(df[raw_col], errors="coerce")
        df_temp = pd.DataFrame({"grid_id": df["grid_id"], "val": v})

        if prefix.startswith("s1_") and prefix.endswith("_mean"):
            new_cols[prefix] = df_temp.groupby("grid_id")["val"].transform("mean")
        else:
            grp = df_temp.groupby("grid_id")["val"]
            mean_series = grp.transform("mean")
            max_series  = grp.transform("max")
            min_series  = grp.transform("min")

            new_cols[f"{prefix}_mean"]  = mean_series
            new_cols[f"{prefix}_max"]   = max_series
            new_cols[f"{prefix}_min"]   = min_series
            new_cols[f"{prefix}_range"] = max_series - min_series

            std_series = grp.transform("std").fillna(0)
            mean_safe  = mean_series.replace(0, float("nan"))
            new_cols[f"{prefix}_cv"]    = (std_series / mean_safe).fillna(0).abs()

    if "ndvi_median_mean" in new_cols:
        gs_mean = new_cols["ndvi_median_mean"].where(
            new_cols["ndvi_median_mean"].fillna(0) > 0.4, other=float("nan")
        ).fillna(new_cols["ndvi_median_mean"])
        new_cols["ndvi_median_growing_season_mean"] = gs_mean

    if new_cols:
        agg_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, agg_df], axis=1)

    return df


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------
def process_region(region: str):
    single_csv_files = []
    
    for search_dir in DATA_SEARCH_DIRS:
        if not search_dir.is_dir():
            continue
        exact_match = search_dir / f"{region}_agri_suitability_with_infra.csv"
        if exact_match.is_file():
            single_csv_files = [exact_match]
            break
        pattern_matches = sorted(search_dir.glob(f"{region}*.csv"))
        if pattern_matches:
            single_csv_files = pattern_matches
            break
        sub_dir = search_dir / region
        if sub_dir.is_dir():
            matches = sorted(sub_dir.glob("*.csv"))
            if matches:
                single_csv_files = matches
                break

    if single_csv_files:
        for raw_csv in single_csv_files:
            print(f"\n  Region: {region.upper()} (Combined Dataset)")
            print(f"    Source File: {raw_csv.name}")

            df = pd.read_csv(raw_csv, low_memory=False)
            df.columns = df.columns.str.strip()

            if "year_month" not in df.columns:
                print(f"    [SKIP] 'year_month' column not found in {raw_csv.name}")
                continue

            print("    Computing feature aggregates across time...")
            if "grid_id" in df.columns:
                df = df.sort_values(by=["grid_id", "year_month"]).reset_index(drop=True)
            df = compute_grid_time_series_aggregates(df)

            grouped = df.groupby("year_month")
            print(f"    Processing {len(grouped)} month slice(s)...")

            for ym_str, month_df in grouped:
                try:
                    year, month = ym_str.split("-")
                except ValueError:
                    continue

                keep = [c for c in KEEP_COLS if c in month_df.columns]
                result = month_df[keep].copy()
                result = result.loc[:, ~result.columns.duplicated()]

                result["region"]     = region
                result["data_year"]  = year
                result["data_month"] = month

                out_dir = PROCESSED_ROOT / region / year / month
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "data.csv"
                result.to_csv(out_path, index=False)

            print(f"    OK -> Output written to {PROCESSED_ROOT / region}")
        return

    # 2. Fall back to static + dynamic files strategy
    static_path = RAW_ROOT / f"{region}_static.csv"
    if not static_path.is_file():
        print(f"  [SKIP] Static file not found: {static_path.name}")
        return

    # Discover dynamic files
    dyn_pattern = re.compile(rf"^{re.escape(region)}_dynamic_(\d{{4}})_(\d{{2}})\.csv$")
    dynamic_files = [
        (f, *dyn_pattern.match(f.name).groups())
        for f in sorted(RAW_ROOT.iterdir())
        if f.is_file() and dyn_pattern.match(f.name)
    ]
    if not dynamic_files:
        print(f"  [SKIP] No dynamic files found for region: {region}")
        return

    print(f"\n  Region: {region.upper()}")
    print(f"    Static : {static_path.name}")
    print(f"    Dynamic: {len(dynamic_files)} file(s)")

    # Load static data
    static_df = pd.read_csv(static_path, low_memory=False)
    static_df.columns = static_df.columns.str.strip()

    for dyn_path, year, month in dynamic_files:
        print(f"    Processing {year}-{month} ...", end=" ")

        dyn_df = pd.read_csv(dyn_path, low_memory=False)
        dyn_df.columns = dyn_df.columns.str.strip()

        merged = pd.merge(
            dyn_df,
            static_df,
            on=JOIN_KEY,
            how="left",
            suffixes=("", "_static"),
        )

        for col in ["latitude", "longitude"]:
            if f"{col}_static" in merged.columns:
                merged[col] = merged[col].fillna(merged[f"{col}_static"])
                merged.drop(columns=[f"{col}_static"], inplace=True)

        merged = merged.loc[:, ~merged.columns.duplicated()]
        merged = compute_aggregates(merged)

        keep = [c for c in KEEP_COLS if c in merged.columns]
        result = merged[keep].copy()
        result = result.loc[:, ~result.columns.duplicated()]

        result["region"]     = region
        result["data_year"]  = year
        result["data_month"] = month

        out_dir = PROCESSED_ROOT / region / year / month
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "data.csv"
        result.to_csv(out_path, index=False)

        print(f"OK  [{len(result)} rows, {len(result.columns)} cols] -> {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw static + dynamic datasets into Parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--region", "-r",
        default=None,
        help=f"Process a single region. Choices: {REGIONS}. Default: all.",
    )
    args = parser.parse_args()

    regions = [args.region] if args.region else REGIONS

    print(f"\n{'='*60}")
    print(f" Myanmar Agricultural ML — Preprocessing Pipeline")
    print(f"{'='*60}")

    for region in regions:
        if region not in REGIONS:
            print(f"[ERROR] Unknown region '{region}'. Valid: {REGIONS}")
            sys.exit(1)
        process_region(region)

    print(f"\n{'='*60}")
    print(f" Preprocessing complete.")
    print(f" Next step: python scripts/label.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
