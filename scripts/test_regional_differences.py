#!/usr/bin/env python3
"""
test_regional_differences.py — Test Regional Suitability & Recommendation Variation
===================================================================================
Tests whether predictions and crop recommendations dynamically change across the
6 target regions (Ayeyawaddy, Bago, Magway, Mandalay, Sagaing, Yangon).
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.core.preprocessor import spatial_manager
from server.core.model_loader import model_manager
from server.services.composite_features import CompositeFeaturesEngine
from server.config import REGIONS, CROPS

REGION_INDICES = {
    "Yangon": "0",
    "Ayeyawaddy": "133056",
    "Bago": "286407",
    "Magway": "439758",
    "Mandalay": "617858",
    "Sagaing": "769427"
}

def main():
    print("=====================================================================")
    print("      TESTING REGIONAL RECOMMENDATION & PREDICTION VARIATION        ")
    print("=====================================================================\n")

    regional_results = {}

    for reg_name, sys_idx in REGION_INDICES.items():
        sample = spatial_manager.lookup_by_system_index(sys_idx)
        if not sample:
            print(f"[ERROR] Failed to find spatial sample for {reg_name} (index {sys_idx})")
            continue

        # Predict crop suitability models for all 17 crops
        predictions = {}
        for crop in CROPS:
            target = f"crop_suitability_{crop}"
            artifact, _ = model_manager.get_model(target)
            if artifact and "model" in artifact:
                model = artifact["model"]
                feats = artifact.get("features", [])
                le = artifact.get("label_encoder")

                x_vals = [float(sample.get(f, 0.0) or 0.0) for f in feats]
                X_in = pd.DataFrame([x_vals], columns=feats)
                pred_raw = model.predict(X_in)[0]

                if le is not None and hasattr(le, "classes_"):
                    pred_label = str(le.classes_[int(pred_raw)])
                else:
                    pred_label = str(pred_raw)

                predictions[target] = pred_label

        # Predict yield & health
        yield_art, _ = model_manager.get_model("crop_yield_t_ha")
        if yield_art:
            X_in = pd.DataFrame([[float(sample.get(f, 0.0) or 0.0) for f in yield_art["features"]]], columns=yield_art["features"])
            predictions["crop_yield_t_ha"] = float(yield_art["model"].predict(X_in)[0])

        health_art, _ = model_manager.get_model("crop_health_score")
        if health_art:
            X_in = pd.DataFrame([[float(sample.get(f, 0.0) or 0.0) for f in health_art["features"]]], columns=health_art["features"])
            predictions["crop_health_score"] = float(health_art["model"].predict(X_in)[0])

        # Get Crop Recommendations
        top_crops = CompositeFeaturesEngine.build_crop_recommender(predictions, top_k=3)
        roi = CompositeFeaturesEngine.build_economic_roi_calculator(predictions)

        regional_results[reg_name] = {
            "top_crops": [f"{c['crop']} ({c['suitability']})" for c in top_crops],
            "yield_t_ha": round(predictions.get("crop_yield_t_ha", 0), 2),
            "health_score": round(predictions.get("crop_health_score", 0), 3),
            "precipitation_mm": round(float(sample.get("chirps_precipitation_mm_mean", 0)), 1),
            "temp_c": round(float(sample.get("mean_temperature_c_mean", 0)), 1)
        }

    print("SUMMARY OF PREDICTIONS PER REGION:\n")
    print(f"{'Region':<12} | {'Precip (mm)':<12} | {'Temp (°C)':<10} | {'Yield (t/ha)':<12} | {'Top 3 Recommended Crops'}")
    print("-" * 80)
    for reg, res in regional_results.items():
        crops_str = ", ".join(res["top_crops"])
        print(f"{reg:<12} | {res['precipitation_mm']:<12} | {res['temp_c']:<10} | {res['yield_t_ha']:<12} | {crops_str}")

    print("\n=====================================================================")

if __name__ == "__main__":
    main()
