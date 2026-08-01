#!/usr/bin/env python3
"""
diagnose_region_features.py — Deep Diagnostic of Regional Features & Model Predictions
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.core.preprocessor import spatial_manager
from server.core.model_loader import model_manager
from server.services.composite_features import CompositeFeaturesEngine
from server.config import CROPS

regions = ["ayeyawaddy", "bago", "magway", "mandalay", "sagaing", "yangon"]

print("=====================================================================")
print("  DIAGNOSING REPRESENTATIVE REGIONAL FEATURES & MODEL PREDICTIONS    ")
print("=====================================================================\n")

for r in regions:
    print(f"--- REGION: {r.upper()} ---")
    subset = spatial_manager.get_region_subset(r)
    print(f"Total points in region: {len(subset):,}")

    if len(subset) == 0:
        print(" [WARN] No points found for region!")
        continue

    # Compare first row vs median row of region
    row_first = subset.iloc[0]
    row_median = subset.iloc[len(subset) // 2]

    for label, row in [("First Sample (iloc[0])", row_first), ("Median Sample", row_median)]:
        precip = row.get("chirps_precipitation_mm_mean", row.get("chirps_precipitation_mm", 0))
        temp = row.get("mean_temperature_c_mean", row.get("mean_temperature_c", 0))
        ndvi = row.get("ndvi_median_growing_season_mean", row.get("ndvi_median_mean", 0))
        crop_pct = row.get("cropland_fraction", 0)

        # Run suitability predictions for all 17 crops
        predictions = {}
        for crop in CROPS:
            target = f"crop_suitability_{crop}"
            artifact, _ = model_manager.get_model(target)
            if artifact and "model" in artifact:
                feats = artifact.get("features", [])
                le = artifact.get("label_encoder")
                X_in = pd.DataFrame([[float(row.get(f, 0.0) or 0.0) for f in feats]], columns=feats)
                pred_raw = artifact["model"].predict(X_in)[0]
                pred_lbl = str(le.classes_[int(pred_raw)]) if le and hasattr(le, "classes_") else str(pred_raw)
                predictions[target] = pred_lbl

        recs = CompositeFeaturesEngine.build_crop_recommender(predictions, top_k=5)
        rec_str = ", ".join([f"{c['crop']}:{c['suitability']}" for c in recs])

        print(f"  {label:<22} | Precip: {precip:6.1f} mm | Temp: {temp:4.1f} °C | NDVI: {ndvi:.4f} | Recs: {rec_str}")

    print()
