#!/usr/bin/env python3
"""
scripts/generate_map_data.py — Generates pre-computed crop recommendations for Myanmar map.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from server.config import CROPS, REGIONS, SUITABILITY_WEIGHTS, SUITABILITY_COLORS
from server.core.model_loader import model_manager

def main():
    print("=====================================================================")
    print("      GENERATING PRE-COMPUTED CROP RECOMMENDATIONS MAP DATA          ")
    print("=====================================================================")

    # 1. Collect points from each region
    processed_dir = PROJECT_ROOT / "data" / "processed"
    all_samples = []

    # We target roughly 400 points per region to get ~2,400 points in total
    TARGET_POINTS_PER_REGION = 400

    for region in REGIONS:
        region_dir = processed_dir / region
        if not region_dir.exists():
            print(f"[WARN] Directory for region {region} not found.")
            continue
        
        # Find the first available monthly data.csv file
        csv_files = list(region_dir.glob("**/data.csv"))
        if not csv_files:
            print(f"[WARN] No data.csv found for region {region}")
            continue
        
        target_csv = csv_files[0]
        print(f"Reading points from {region} ({target_csv.relative_to(PROJECT_ROOT)})...")
        
        df = pd.read_csv(target_csv)
        if df.empty:
            continue
            
        # Ensure we have coordinates and ID
        if "latitude" not in df.columns or "longitude" not in df.columns:
            print(f"[WARN] Lat/lon missing in {region} csv")
            continue
            
        # Downsample uniformly
        n_samples = len(df)
        step = max(1, n_samples // TARGET_POINTS_PER_REGION)
        downsampled_df = df.iloc[::step].copy()
        
        # Add region info to downsampled dataframe
        downsampled_df["region_name"] = region
        for r in REGIONS:
            downsampled_df[f"region_{r}"] = 1 if r == region else 0
            
        # Add crop_area_pct_* columns
        from scripts.label import add_regional_crop_pct_features
        downsampled_df = add_regional_crop_pct_features(downsampled_df)
            
        all_samples.append(downsampled_df)

    if not all_samples:
        print("[ERROR] No sample points could be collected. Exiting.")
        sys.exit(1)
        
    combined_df = pd.concat(all_samples, ignore_index=True)
    n_points = len(combined_df)
    print(f"Collected {n_points} total downsampled points across all regions.")

    # 2. Batch predict crop suitabilities
    print("Pre-loading models and batch predicting crop suitabilities...")
    
    # Store predictions for each crop target
    crop_preds = {}
    
    for crop in CROPS:
        target = f"crop_suitability_{crop}"
        model_artifact, source = model_manager.get_model(target, force_prototype=False)
        if not model_artifact or "model" not in model_artifact:
            print(f"[WARN] Failed to load model for crop {crop}")
            continue
            
        model = model_artifact["model"]
        features = model_artifact["features"]
        le = model_artifact.get("label_encoder")
        
        # Prepare feature matrix for this model
        X = combined_df[features].copy()
        # Handle NaN values
        X = X.fillna(X.median())
        
        # Predict in batch
        preds_num = model.predict(X)
        if le is not None:
            preds_label = le.inverse_transform(preds_num)
        else:
            preds_label = preds_num
            
        crop_preds[crop] = [str(lbl).lower() for lbl in preds_label]
        print(f"  ✓ Predicted suitability for crop: {crop}")

    # 3. Calculate crop rankings for each point
    print("Ranking crop recommendations for all points...")
    map_points = []
    
    for i in range(n_points):
        row = combined_df.iloc[i]
        
        # Calculate suitability score for each crop for this point
        crop_scores = []
        for crop in CROPS:
            if crop not in crop_preds:
                continue
                
            lbl = crop_preds[crop][i]
            weight = SUITABILITY_WEIGHTS.get(lbl, 0.40)
            suitability_pct = round(weight * 100.0, 1)
            
            # Regional historical area tie-breaker
            area_pct = float(row.get(f"crop_area_pct_{crop}", 0.0) or 0.0)
            composite_score = suitability_pct + (area_pct * 0.1)
            
            crop_scores.append({
                "crop": crop,
                "suitability": lbl,
                "suitability_score": suitability_pct,
                "composite_score": composite_score,
                "color_code": SUITABILITY_COLORS.get(lbl, "#3B82F6")
            })
            
        # Sort crop recommendations
        crop_scores.sort(key=lambda x: (x["suitability_score"], x["composite_score"]), reverse=True)
        
        # Adjust suitability_score to ensure strictly descending order
        for idx, item in enumerate(crop_scores):
            base_score = item["suitability_score"]
            area_pct = float(row.get(f"crop_area_pct_{item['crop']}", 0.0) or 0.0)
            adjusted_score = base_score + (area_pct * 0.05) - (idx * 0.1)
            item["suitability_score"] = round(max(1.0, min(100.0, adjusted_score)), 1)
        
        # Store metadata with compact [crop, score] recommendations to minimize payload size.
        # Client derives suitability label and colors from the score value.
        compact_recs = [[item["crop"], item["suitability_score"]] for item in crop_scores]
        
        map_points.append({
            "index": str(row.get("system:index", i)),
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "region": str(row["region_name"]),
            "recommendations": compact_recs
        })

    # Save to server/static/map_recommendations.json
    out_dir = PROJECT_ROOT / "server" / "static"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "map_recommendations.json"
    
    with open(out_file, "w") as f:
        json.dump(map_points, f, separators=(",", ":"))  # no whitespace = smallest possible size
        
    print(f"\n[SUCCESS] Successfully pre-computed crop recommendations and saved to {out_file}")
    print(f"File size: {out_file.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
