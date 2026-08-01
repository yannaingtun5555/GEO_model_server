#!/usr/bin/env python3
"""
recommend_crops.py — Crop Recommendation & Region Suitability Analyzer
========================================================================
Loads all 11 crop suitability classification models and evaluates which crops
are best suited to plant for each region (Ayeyawaddy, Bago, Magway, Mandalay, Sagaing).

Usage:
    python scripts/recommend_crops.py
    python scripts/recommend_crops.py --region ayeyawaddy
    python scripts/recommend_crops.py --top-k 3
    python scripts/recommend_crops.py --data data/combined/combined_dataset.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE    = PROJECT_ROOT / "data" / "combined" / "combined_dataset.csv"
MODELS_DIR   = PROJECT_ROOT / "models"

CROPS = [
    "monsoon_rice",
    "dry_season_rice",
    "maize",
    "sugarcane",
    "cassava",
    "durian",
    "mangosteen",
    "longan",
    "mango",
    "chili",
    "tomato",
    "black_gram",
    "green_gram",
    "pigeon_pea",
    "groundnut",
    "sesame",
    "rubber",
]

REGIONS = ["ayeyawaddy", "bago", "magway", "mandalay", "sagaing", "yangon"]

# Suitability rank weights for scoring
SUITABILITY_WEIGHTS = {
    "excellent": 1.0,
    "good": 0.75,
    "moderate": 0.40,
    "poor": 0.0,
}


def load_dependencies():
    try:
        import joblib
        return {"joblib": joblib}
    except ImportError:
        print("[ERROR] Missing joblib dependency. Run: pip install joblib scikit-learn pandas")
        sys.exit(1)


def load_crop_models(models_dir: Path, sk: dict) -> dict:
    """Load trained models for all 11 crop targets."""
    loaded_models = {}
    for crop in CROPS:
        model_path = models_dir / f"crop_suitability_{crop}_rf_classifier.pkl"
        if not model_path.exists():
            print(f"[WARN] Model artifact missing for crop '{crop}': {model_path.name}")
            continue

        try:
            artifact = sk["joblib"].load(model_path)
            loaded_models[crop] = artifact
        except Exception as e:
            print(f"[ERROR] Failed to load model for {crop}: {e}")

    return loaded_models


def analyze_region_recommendations(df: pd.DataFrame, models: dict, region_filter: str = None, top_k: int = 3):
    """Predict crop suitability for all rows and aggregate recommendations per region."""
    results = []
    grid_recommendations = []

    # Detect region column format
    region_cols = [c for c in df.columns if c.startswith("region_")]

    for r_name in REGIONS:
        if region_filter and region_filter.lower() != r_name:
            continue

        oh_col = f"region_{r_name}"
        if oh_col in df.columns:
            region_df = df[df[oh_col] == 1].copy()
        elif "region" in df.columns:
            region_df = df[df["region"].astype(str).str.lower() == r_name].copy()
        else:
            region_df = df.copy()

        if len(region_df) == 0:
            continue

        crop_scores = {}
        crop_counts = {}

        for crop, artifact in models.items():
            model = artifact["model"]
            features = artifact["features"]
            le = artifact.get("label_encoder")

            missing_feats = [f for f in features if f not in region_df.columns]
            if missing_feats:
                continue

            X = region_df[features]

            # Handle NaN filling with median
            X = X.fillna(X.median())

            preds_num = model.predict(X)
            if le is not None:
                preds_label = le.inverse_transform(preds_num)
            else:
                preds_label = preds_num

            # Calculate weighted suitability index (0.0 to 100.0%)
            weighted_score = sum(SUITABILITY_WEIGHTS.get(str(lbl).lower(), 0.0) for lbl in preds_label) / len(preds_label) * 100.0

            # Count distribution of categories
            counts = pd.Series(preds_label).value_counts().to_dict()
            excellent_pct = (counts.get("excellent", 0) / len(preds_label)) * 100.0
            good_pct = (counts.get("good", 0) / len(preds_label)) * 100.0
            moderate_pct = (counts.get("moderate", 0) / len(preds_label)) * 100.0
            poor_pct = (counts.get("poor", 0) / len(preds_label)) * 100.0

            crop_scores[crop] = weighted_score
            crop_counts[crop] = {
                "weighted_score": weighted_score,
                "excellent_pct": excellent_pct,
                "good_pct": good_pct,
                "suitable_pct": excellent_pct + good_pct,
                "moderate_pct": moderate_pct,
                "poor_pct": poor_pct,
            }

        # Rank crops by suitability score
        sorted_crops = sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)

        rank_str = ", ".join([f"{c[0]} ({c[1]:.1f}%)" for c in sorted_crops[:top_k]])

        for rank, (crop_name, score) in enumerate(sorted_crops, 1):
            stats = crop_counts[crop_name]
            results.append({
                "region": r_name.capitalize(),
                "rank": rank,
                "crop": crop_name,
                "suitability_score": round(score, 2),
                "excellent_pct": round(stats["excellent_pct"], 2),
                "good_pct": round(stats["good_pct"], 2),
                "suitable_pct": round(stats["suitable_pct"], 2),
                "moderate_pct": round(stats["moderate_pct"], 2),
                "poor_pct": round(stats["poor_pct"], 2),
                "total_samples": len(region_df),
            })

        print(f"\n📍 Region: {r_name.upper()} ({len(region_df):,} samples)")
        print(f"   Top {top_k} Recommended Crops to Plant:")
        for idx, (crop_name, score) in enumerate(sorted_crops[:top_k], 1):
            stats = crop_counts[crop_name]
            print(f"   {idx}. {crop_name:<20} | Suitability Index: {score:.1f}% | (Excellent: {stats['excellent_pct']:.1f}%, Good: {stats['good_pct']:.1f}%)")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Recommend Best Crops to Plant Per Region")
    parser.add_argument("--data", type=str, default=str(DATA_FILE), help="Path to combined dataset CSV")
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR), help="Path to models directory")
    parser.add_argument("--region", type=str, default=None, help="Filter by region (e.g. ayeyawaddy, bago)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top crops to show per region")
    args = parser.parse_args()

    sk = load_dependencies()
    data_path = Path(args.data)
    models_dir = Path(args.models_dir)

    print("=====================================================================")
    print("      CROP RECOMMENDATION & REGION SUITABILITY ANALYSIS              ")
    print("=====================================================================")
    print(f" Dataset Path : {data_path}")
    print(f" Models Dir   : {models_dir}")
    if args.region:
        print(f" Region Filter: {args.region.capitalize()}")
    print("---------------------------------------------------------------------\n")

    if not data_path.exists():
        print(f"[ERROR] Dataset file not found: {data_path}")
        sys.exit(1)

    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df):,} rows.\n")

    models = load_crop_models(models_dir, sk)
    if not models:
        print("[ERROR] No crop suitability models found. Run ./run.sh train first.")
        sys.exit(1)

    print(f"Loaded {len(models)} crop suitability model(s).\n")

    rec_df = analyze_region_recommendations(df, models, region_filter=args.region, top_k=args.top_k)

    if not rec_df.empty:
        out_csv = models_dir / "region_crop_recommendations.csv"
        rec_df.to_csv(out_csv, index=False)
        print(f"\n[SUMMARY] Saved region crop recommendations to: {out_csv}\n")


if __name__ == "__main__":
    main()
