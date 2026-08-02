#!/usr/bin/env python3
"""
scripts/test_seasonal_changes.py — Evaluates model predictions across different seasons
=====================================================================================
Selects representative grid points from the dataset and simulates two extreme seasons:
1. Wet/Monsoon Season (August): High precipitation, high soil moisture, high NDVI/NDWI.
2. Dry Season (January): Low precipitation, low soil moisture, lower NDVI/NDWI.

Runs batch predictions and compares how crop suitability, drought risk, flood risk,
and irrigation need change in response to seasonal dynamics.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from server.config import CROPS, REGIONS
from server.core.model_loader import model_manager

def simulate_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Modifies dynamic features in the dataframe to simulate a given season."""
    df_sim = df.copy()
    
    if season == "monsoon":
        df_sim["data_month"] = 8  # August
        # Simulating monsoon values
        df_sim["chirps_precipitation_mm"] = 350.0
        df_sim["chirps_precipitation_mm_mean"] = 300.0
        df_sim["mean_temperature_c"] = 26.5
        df_sim["mean_temperature_c_mean"] = 27.0
        df_sim["era5_soil_moisture_m3_m3_mean"] = 0.45
        df_sim["ndvi_median_mean"] = 0.75
        df_sim["ndvi_median_growing_season_mean"] = 0.72
        df_sim["ndwi_mcf_median_mean"] = 0.35
        df_sim["solar_radiation_mj_m2_day"] = 14.5
        
    elif season == "dry":
        df_sim["data_month"] = 1  # January
        # Simulating dry season values
        df_sim["chirps_precipitation_mm"] = 5.0
        df_sim["chirps_precipitation_mm_mean"] = 10.0
        df_sim["mean_temperature_c"] = 24.0
        df_sim["mean_temperature_c_mean"] = 24.5
        df_sim["era5_soil_moisture_m3_m3_mean"] = 0.18
        df_sim["ndvi_median_mean"] = 0.42
        df_sim["ndvi_median_growing_season_mean"] = 0.50
        df_sim["ndwi_mcf_median_mean"] = -0.15
        df_sim["solar_radiation_mj_m2_day"] = 21.0
        
    return df_sim

def main():
    print("=====================================================================")
    print("      TESTING MODEL PREDICTION SENSITIVITY TO SEASONAL CHANGES        ")
    print("=====================================================================")

    # 1. Collect one representative point from each region
    processed_dir = PROJECT_ROOT / "data" / "processed"
    sample_rows = []
    
    for region in REGIONS:
        region_dir = processed_dir / region
        csv_files = list(region_dir.glob("**/data.csv"))
        if not csv_files:
            continue
        
        df = pd.read_csv(csv_files[0])
        if not df.empty:
            row = df.iloc[0].copy()
            row["region_name"] = region
            for r in REGIONS:
                row[f"region_{r}"] = 1 if r == region else 0
            sample_rows.append(row)
            
    if not sample_rows:
        print("[ERROR] No representative region points found.")
        sys.exit(1)
        
    points_df = pd.DataFrame(sample_rows)
    from scripts.label import add_regional_crop_pct_features
    points_df = add_regional_crop_pct_features(points_df)
    print(f"Loaded {len(points_df)} representative grid points (one per region).")

    # Dynamic features to print
    dynamic_features = [
        "data_month", "chirps_precipitation_mm", "mean_temperature_c",
        "era5_soil_moisture_m3_m3_mean", "ndvi_median_mean", "ndwi_mcf_median_mean"
    ]

    # Target variables of interest
    test_targets = [
        "crop_suitability_monsoon_rice",
        "crop_suitability_dry_season_rice",
        "crop_suitability_maize",
        "crop_suitability_chili",
        "drought_risk_score",
        "flood_risk_level",
        "irrigation_need"
    ]

    # Apply seasonal simulations
    monsoon_df = simulate_season(points_df, "monsoon")
    dry_df = simulate_season(points_df, "dry")
    
    # Run predictions for each season
    results = {}
    
    for season_name, df_season in [("Monsoon (August)", monsoon_df), ("Dry Season (January)", dry_df)]:
        print(f"\nEvaluating targets for {season_name}...")
        season_preds = {}
        
        for target in test_targets:
            model_artifact, _ = model_manager.get_model(target, force_prototype=False)
            if not model_artifact or "model" not in model_artifact:
                print(f"  [WARN] Model for {target} not available.")
                continue
                
            model = model_artifact["model"]
            features = model_artifact["features"]
            le = model_artifact.get("label_encoder")
            
            # Predict
            X = df_season[features].copy()
            X = X.fillna(X.median())
            preds_num = model.predict(X)
            
            if le is not None:
                preds_label = le.inverse_transform(preds_num)
                season_preds[target] = [str(lbl).upper() for lbl in preds_label]
            else:
                season_preds[target] = [round(float(v), 4) for v in preds_num]
                
        results[season_name] = season_preds

    # 3. Print Comparison Report
    report_file = PROJECT_ROOT / "models" / "seasonal_changes_report.txt"
    report_lines = []
    
    def r_print(msg):
        print(msg)
        report_lines.append(msg)

    r_print("=====================================================================")
    r_print("                  SEASONAL SENSITIVITY REPORT                        ")
    r_print("=====================================================================")
    
    for i, region in enumerate(points_df["region_name"].values):
        r_print(f"\n📍 Region: {region.upper()} (Lat: {points_df.iloc[i]['latitude']:.4f}, Lon: {points_df.iloc[i]['longitude']:.4f})")
        r_print("-" * 70)
        
        # Print table header
        r_print(f"{'Target Variable':35s} | {'Monsoon (August)':18s} | {'Dry Season (January)':20s}")
        r_print("-" * 70)
        
        for target in test_targets:
            monsoon_val = results["Monsoon (August)"][target][i]
            dry_val = results["Dry Season (January)"][target][i]
            
            # Highlight changes
            marker = "  "
            if monsoon_val != dry_val:
                marker = "⚡"
                
            target_clean = target.replace("crop_suitability_", "suitability_")
            r_print(f"{marker} {target_clean:32s} | {str(monsoon_val):18s} | {str(dry_val):20s}")
            
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))
        
    print(f"\n[SUCCESS] Saved seasonal sensitivity report to: {report_file}")

if __name__ == "__main__":
    main()
