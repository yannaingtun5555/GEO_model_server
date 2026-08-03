#!/usr/bin/env python3
"""
server/services/gee_pipeline.py — Live GEE Ingestion & Batch Prediction Pipeline
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

from server.config import PROJECT_ROOT, CROPS, SUITABILITY_WEIGHTS, SUITABILITY_COLORS
from server.core.model_loader import model_manager
from server.core.preprocessor import spatial_manager
from server.core.live_prediction_manager import live_prediction_manager, LIVE_PREDICTIONS_PARQUET

# List of 40 targets we need to predict
ALL_TARGETS = [
    "crop_health_score", "crop_yield_t_ha", "irrigation_need",
    "current_month_precipitation_mm", "current_month_mean_temperature_c", "current_month_solar_rad_mj_m2_day",
    "flood_risk_level", "drought_risk_score", "heat_stress_risk", "optimal_planting_month",
    "nitrogen_requirement_level", "phosphorus_requirement_level", "soil_erosion_risk",
    "market_integration_score", "post_harvest_loss_risk", "supply_chain_efficiency",
    "cold_chain_potential", "agricultural_land_conversion_risk", "urban_encroachment_risk",
    "irrigation_potential", "surface_water_occurrence", "water_scarcity_risk", "agricultural_gdp_forecast"
] + [f"crop_suitability_{c}" for c in CROPS]

def run_gee_ingestion_pipeline(update_csv_path: Path) -> dict:
    """
    Ingests dynamic GEE update CSV, merges it with static features,
    runs batch predictions for all 40 outputs, saves predictions,
    regenerates map recommendations, and reloads server cache.
    """
    print(f"[GEE PIPELINE] Starting GEE Ingestion pipeline with file: {update_csv_path}")
    
    # 1. Load uploaded GEE updates
    df_update = pd.read_csv(update_csv_path)
    if "system:index" not in df_update.columns:
        # Fallback to indexing column if system:index is missing
        if "index" in df_update.columns:
            df_update = df_update.rename(columns={"index": "system:index"})
        else:
            raise ValueError("Uploaded GEE update CSV must contain a 'system:index' or 'index' column.")
            
    df_update["system:index"] = df_update["system:index"].astype(str)
    
    # 2. Get the master features dataset
    if spatial_manager.df is None:
        spatial_manager._load_dataset()
    if spatial_manager.df is None:
        raise ValueError("Master spatial features dataset could not be loaded.")
        
    master_df = spatial_manager.df.copy()
    
    # 3. Add coords back to features dataset if they are loaded from processed CSVs
    # We load them from processed datasets to keep coords for predictions dataset
    # We will use coordinates from the master_df index mapping
    coords_df = pd.DataFrame(columns=["system:index", "latitude", "longitude"])
    
    # Check if coords are in master_df. If not, try to read from a processed file to match system:index
    if "latitude" not in master_df.columns:
        processed_dir = PROJECT_ROOT / "data" / "processed"
        csv_files = list(processed_dir.glob("**/data.csv"))
        if csv_files:
            try:
                raw_coords = pd.read_csv(csv_files[0], usecols=["system:index", "latitude", "longitude"])
                raw_coords["system:index"] = raw_coords["system:index"].astype(str)
                coords_map = raw_coords.set_index("system:index")[["latitude", "longitude"]].to_dict("index")
            except Exception:
                coords_map = {}
        else:
            coords_map = {}
    else:
        coords_map = master_df.set_index("system:index")[["latitude", "longitude"]].to_dict("index")

    # 4. Merge GEE updates into master features dataset
    # For matching system:index rows, overwrite dynamic columns with new values
    dynamic_cols = [c for c in df_update.columns if c in master_df.columns and c != "system:index"]
    print(f"[GEE PIPELINE] Overwriting {len(dynamic_cols)} dynamic columns for matching points: {dynamic_cols}")
    
    df_update_set = df_update.set_index("system:index")
    master_df_set = master_df.set_index("system:index")
    
    # Overwrite the intersecting indices
    common_idx = master_df_set.index.intersection(df_update_set.index)
    if len(common_idx) > 0:
        master_df_set.loc[common_idx, dynamic_cols] = df_update_set.loc[common_idx, dynamic_cols]
    
    # Reset index
    updated_features_df = master_df_set.reset_index()
    
    # Add crop area percentage features (label helper)
    from scripts.label import add_regional_crop_pct_features
    # Check if 'region' column is present, if not construct it from one-hot cols
    if "region" not in updated_features_df.columns:
        regions = ["ayeyawaddy", "bago", "magway", "mandalay", "sagaing", "yangon"]
        region_series = pd.Series("", index=updated_features_df.index)
        for r in regions:
            if f"region_{r}" in updated_features_df.columns:
                region_series = np.where(updated_features_df[f"region_{r}"] == 1, r, region_series)
        updated_features_df["region"] = region_series
        
    updated_features_df = add_regional_crop_pct_features(updated_features_df)

    # 5. Run Batch Predictions for all 40 targets
    print(f"[GEE PIPELINE] Running batch predictions for {len(updated_features_df):,} points...")
    predictions_df = pd.DataFrame()
    predictions_df["system:index"] = updated_features_df["system:index"]
    
    # Map lat/lon for lookup database
    latitudes = []
    longitudes = []
    region_names = []
    
    for idx in updated_features_df["system:index"].values:
        idx_str = str(idx)
        pt = coords_map.get(idx_str, {"latitude": 21.0, "longitude": 96.0})
        latitudes.append(pt.get("latitude", 21.0))
        longitudes.append(pt.get("longitude", 96.0))
        
        # Determine region
        # Fallback region finding
        r_name = "yangon"
        for r in ["ayeyawaddy", "bago", "magway", "mandalay", "sagaing", "yangon"]:
            if f"region_{r}" in updated_features_df.columns:
                row_match = updated_features_df[updated_features_df["system:index"] == idx]
                if not row_match.empty and row_match.iloc[0][f"region_{r}"] == 1:
                    r_name = r
                    break
        region_names.append(r_name)
        
    predictions_df["latitude"] = latitudes
    predictions_df["longitude"] = longitudes
    predictions_df["region"] = region_names

    # Run predictions target by target
    for target in ALL_TARGETS:
        model_artifact, source = model_manager.get_model(target, force_prototype=False)
        if not model_artifact or "model" not in model_artifact:
            print(f"[GEE PIPELINE] Model missing for {target}, skipping.")
            continue
            
        model = model_artifact["model"]
        features = model_artifact["features"]
        le = model_artifact.get("label_encoder")
        
        # Predict batch
        X = updated_features_df[features].copy()
        X = X.fillna(X.median())
        
        preds_num = model.predict(X)
        if le is not None:
            preds_label = le.inverse_transform(preds_num)
            predictions_df[target] = [str(lbl).lower() for lbl in preds_label]
        else:
            predictions_df[target] = [float(val) for val in preds_num]
            
        print(f"  ✓ Batch predicted target: {target}")

    # 6. Save prediction database
    LIVE_PREDICTIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(LIVE_PREDICTIONS_PARQUET, index=False)
    predictions_df.to_csv(LIVE_PREDICTIONS_CSV, index=False)
    
    print(f"[GEE PIPELINE] Saved prediction database to {LIVE_PREDICTIONS_PARQUET} ({LIVE_PREDICTIONS_PARQUET.stat().st_size/1024:.1f} KB)")

    # 7. Re-generate map_recommendations.json with ALL 17 crops
    print("[GEE PIPELINE] Re-generating map recommendations for frontend...")
    
    # We downsample predictions_df for the map interface (~2,500 points)
    TARGET_POINTS = 2500
    step = max(1, len(predictions_df) // TARGET_POINTS)
    map_df = predictions_df.iloc[::step].copy()
    
    map_points = []
    for i in range(len(map_df)):
        row = map_df.iloc[i]
        
        # Calculate suitability score for all crops
        crop_scores = []
        for crop in CROPS:
            target = f"crop_suitability_{crop}"
            if target not in row:
                continue
            lbl = str(row[target]).lower()
            weight = SUITABILITY_WEIGHTS.get(lbl, 0.40)
            suitability_pct = round(weight * 100.0, 1)
            
            # Look up crop area pct from the feature matrix
            feat_row = updated_features_df[updated_features_df["system:index"] == row["system:index"]]
            area_pct = 0.0
            if not feat_row.empty:
                area_pct = float(feat_row.iloc[0].get(f"crop_area_pct_{crop}", 0.0) or 0.0)
                
            composite_score = suitability_pct + (area_pct * 0.1)
            
            crop_scores.append({
                "crop": crop,
                "suitability": lbl,
                "suitability_score": suitability_pct,
                "composite_score": composite_score,
                "color_code": SUITABILITY_COLORS.get(lbl, "#3B82F6")
            })
            
        # Sort crop recommendations descending by score
        crop_scores.sort(key=lambda x: (x["suitability_score"], x["composite_score"]), reverse=True)
        
        # Adjust suitability_score to ensure strictly descending order
        for idx, item in enumerate(crop_scores):
            base_score = item["suitability_score"]
            # Look up crop area pct from the feature matrix
            feat_row = updated_features_df[updated_features_df["system:index"] == row["system:index"]]
            area_pct = 0.0
            if not feat_row.empty:
                area_pct = float(feat_row.iloc[0].get(f"crop_area_pct_{item['crop']}", 0.0) or 0.0)
            adjusted_score = base_score + (area_pct * 0.05) - (idx * 0.1)
            item["suitability_score"] = round(max(1.0, min(100.0, adjusted_score)), 1)
        
        # Store compact [crop, score] recommendations — client derives labels/colors from score
        compact_recs = [[item["crop"], item["suitability_score"]] for item in crop_scores]
        map_points.append({
            "index": str(row["system:index"]),
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "region": str(row["region"]),
            "recommendations": compact_recs
        })
        
    out_json = PROJECT_ROOT / "server" / "static" / "map_recommendations.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(map_points, f, separators=(",", ":"))  # no whitespace = smallest possible size
        
    print(f"[GEE PIPELINE] Saved map recommendations to {out_json}")

    # 8. Reload live prediction manager cache
    live_prediction_manager.load_predictions()
    
    return {
        "status": "success",
        "points_updated": len(common_idx),
        "total_points_predicted": len(predictions_df),
        "predictions_file_size_kb": round(LIVE_PREDICTIONS_PARQUET.stat().st_size / 1024.0, 1)
    }
