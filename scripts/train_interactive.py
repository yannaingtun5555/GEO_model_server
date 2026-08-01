#!/usr/bin/env python3
"""Interactive Training - Full Model (500 trees, 5-fold CV)"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import gc
import time
import json
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# TARGETS DEFINITION
# ─────────────────────────────────────────────────────────────
SUITABILITY_LABELS = ["poor", "moderate", "good", "excellent"]

TARGETS = {
    "crop_suitability_monsoon_rice": {'type': 'classification', 'importance': 'high'},
    "crop_suitability_dry_season_rice": {'type': 'classification', 'importance': 'high'},
    "crop_suitability_black_gram": {'type': 'classification', 'importance': 'high'},
    "crop_suitability_groundnut": {'type': 'classification', 'importance': 'high'},
    "crop_health_score": {'type': 'regression', 'importance': 'high'},
    "crop_yield_t_ha": {'type': 'regression', 'importance': 'high'},
    "irrigation_need": {'type': 'classification', 'importance': 'high'},
    "flood_risk_level": {'type': 'classification', 'importance': 'high'},
    "drought_risk_score": {'type': 'regression', 'importance': 'high'},
    "heat_stress_risk": {'type': 'classification', 'importance': 'high'},
    "crop_suitability_maize": {'type': 'classification', 'importance': 'medium'},
    "crop_suitability_sugarcane": {'type': 'classification', 'importance': 'medium'},
    "crop_suitability_cassava": {'type': 'classification', 'importance': 'medium'},
    "crop_suitability_chili": {'type': 'classification', 'importance': 'medium'},
    "crop_suitability_tomato": {'type': 'classification', 'importance': 'medium'},
    "crop_suitability_green_gram": {'type': 'classification', 'importance': 'medium'},
    "crop_suitability_pigeon_pea": {'type': 'classification', 'importance': 'medium'},
    "crop_suitability_sesame": {'type': 'classification', 'importance': 'medium'},
    "crop_suitability_rubber": {'type': 'classification', 'importance': 'medium'},
    "current_month_precipitation_mm": {'type': 'regression', 'importance': 'medium'},
    "current_month_mean_temperature_c": {'type': 'regression', 'importance': 'medium'},
    "current_month_solar_rad_mj_m2_day": {'type': 'regression', 'importance': 'medium'},
    "optimal_planting_month": {'type': 'classification', 'importance': 'medium'},
    "nitrogen_requirement_level": {'type': 'classification', 'importance': 'medium'},
    "phosphorus_requirement_level": {'type': 'classification', 'importance': 'medium'},
    "soil_erosion_risk": {'type': 'classification', 'importance': 'medium'},
    "market_integration_score": {'type': 'regression', 'importance': 'medium'},
    "post_harvest_loss_risk": {'type': 'regression', 'importance': 'medium'},
    "supply_chain_efficiency": {'type': 'regression', 'importance': 'medium'},
    "cold_chain_potential": {'type': 'regression', 'importance': 'medium'},
    "agricultural_land_conversion_risk": {'type': 'regression', 'importance': 'medium'},
    "urban_encroachment_risk": {'type': 'regression', 'importance': 'medium'},
    "irrigation_potential": {'type': 'regression', 'importance': 'medium'},
    "surface_water_occurrence": {'type': 'regression', 'importance': 'medium'},
    "water_scarcity_risk": {'type': 'regression', 'importance': 'medium'},
    "agricultural_gdp_forecast": {'type': 'regression', 'importance': 'high'},
    "crop_suitability_durian": {'type': 'classification', 'importance': 'low'},
    "crop_suitability_mangosteen": {'type': 'classification', 'importance': 'low'},
    "crop_suitability_longan": {'type': 'classification', 'importance': 'low'},
    "crop_suitability_mango": {'type': 'classification', 'importance': 'low'},
}

# ─────────────────────────────────────────────────────────────
# Feature Weights
# ─────────────────────────────────────────────────────────────
FEATURE_WEIGHTS = {
    "soil_soc_g_kg_0_30cm": 3.0,
    "soil_cec_cmol_kg_0_30cm": 2.5,
    "soil_ph_h2o_0_30cm": 2.5,
    "soil_clay_pct_0_30cm": 1.5,
    "soil_sand_pct_0_30cm": 1.5,
    "soil_silt_pct_0_30cm": 1.2,
    "era5_soil_moisture_m3_m3_mean": 3.0,
    "era5_soil_moisture_m3_m3_min": 2.5,
    "era5_soil_moisture_m3_m3_cv": 1.8,
    "ndwi_mcf_median_mean": 2.0,
    "ndwi_mcf_median_max": 1.5,
    "distance_to_surface_water_m": 1.8,
    "surface_water_occurrence_pct": 1.5,
    "surface_water_seasonality_months": 1.3,
    "chirps_precipitation_mm_mean": 2.5,
    "chirps_precipitation_mm_min": 2.0,
    "chirps_precipitation_mm_cv": 1.8,
    "chirps_precipitation_mm_max": 1.5,
    "chirps_precipitation_mm_range": 1.3,
    "chirps_precipitation_mm": 1.5,
    "ndvi_median_mean": 2.5,
    "ndvi_median_growing_season_mean": 2.5,
    "ndvi_median_max": 2.0,
    "ndvi_median_min": 1.5,
    "mean_temperature_c_mean": 2.0,
    "mean_temperature_c_max": 1.8,
    "mean_temperature_c_min": 1.8,
    "mean_temperature_c_range": 1.5,
    "mean_temperature_c": 1.5,
    "solar_radiation_mj_m2_day_mean": 1.8,
    "solar_radiation_mj_m2_day_max": 1.5,
    "solar_radiation_mj_m2_day": 1.5,
    "elevation_m": 1.8,
    "slope_degrees": 1.5,
    "aspect_degrees": 1.0,
    "s1_vh_db_median_mean": 1.3,
    "s1_vv_db_median_mean": 1.3,
    "data_month": 1.5,
    "crop_area_pct_monsoon_rice": 3.5,
    "crop_area_pct_dry_season_rice": 3.0,
    "crop_area_pct_sesame": 2.5,
    "crop_area_pct_groundnut": 2.5,
    "crop_area_pct_black_gram": 2.5,
    "crop_area_pct_green_gram": 2.5,
    "crop_area_pct_pigeon_pea": 2.5,
    "crop_area_pct_maize": 2.0,
    "crop_area_pct_sugarcane": 2.0,
    "crop_area_pct_cassava": 2.0,
}

# ─────────────────────────────────────────────────────────────
# FULL TRAINING PARAMETERS (500 trees, 5-fold CV)
# ─────────────────────────────────────────────────────────────
FULL_PARAMS = {
    "clf": dict(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),
    "reg": dict(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),
}

def load_sklearn():
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder
    import joblib
    return {
        "RFC": RandomForestClassifier, "RFR": RandomForestRegressor,
        "tts": train_test_split, "SKF": StratifiedKFold, "KF": KFold,
        "cvs": cross_val_score, "acc": accuracy_score, "f1": f1_score,
        "mae": mean_absolute_error, "mse": mean_squared_error, "r2": r2_score,
        "LE": LabelEncoder, "jl": joblib,
    }

def build_sample_weights(X, feature_weights):
    w = np.ones(len(X))
    for col, fw in feature_weights.items():
        if col in X.columns and fw > 1.0:
            series = pd.to_numeric(X[col], errors="coerce").fillna(0)
            rng = series.max() - series.min()
            if rng > 0:
                norm = (series - series.min()) / rng
                w = w + norm.values * (fw - 1.0)
    w = w / w.mean()
    return w.clip(0.5, 3.0)

def train_classifier(target, df, feature_cols, params, sk, out_dir):
    info = TARGETS[target]
    y_raw = df[target].dropna()
    X_raw = df.loc[y_raw.index, feature_cols]
    
    le = sk["LE"]()
    if info.get("classes") and isinstance(info["classes"][0], str):
        le.fit(info["classes"])
    else:
        le.fit(sorted(y_raw.unique()))
    y = le.transform(y_raw.astype(str))
    sample_w = build_sample_weights(X_raw, FEATURE_WEIGHTS)
    
    pbar = tqdm(total=100, desc=f"  Training {target}", leave=False, position=0)
    pbar.update(10)
    
    try:
        X_tr, X_te, y_tr, y_te, sw_tr, _ = sk["tts"](
            X_raw, y, sample_w, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te, sw_tr, _ = sk["tts"](
            X_raw, y, sample_w, test_size=0.2, random_state=42
        )
    
    pbar.update(20)
    
    p = params["clf"].copy()
    p.pop("class_weight", None)
    model = sk["RFC"](**p, class_weight="balanced")
    model.fit(X_tr, y_tr, sample_weight=sw_tr)
    pbar.update(40)
    
    y_pred = model.predict(X_te)
    acc = sk["acc"](y_te, y_pred)
    f1_mac = sk["f1"](y_te, y_pred, average="macro", zero_division=0)
    pbar.update(10)
    
    kf = sk["SKF"](n_splits=5, shuffle=True, random_state=42)
    cv_scores = sk["cvs"](model, X_raw, y, cv=kf, scoring="f1_macro", n_jobs=-1)
    pbar.update(10)
    
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    model_path = out_dir / f"{target}_rf_classifier.pkl"
    sk["jl"].dump({"model": model, "label_encoder": le, "features": feature_cols}, model_path)
    fi.to_csv(out_dir / f"fi_{target}.csv", index=False)
    
    pbar.update(10)
    pbar.close()
    
    return {
        "target": target, "type": "classification",
        "n_train": len(X_tr), "n_test": len(X_te),
        "accuracy": round(acc, 4), "f1_macro": round(f1_mac, 4),
        "cv_f1_mean": round(float(np.mean(cv_scores)), 4),
        "cv_f1_std": round(float(np.std(cv_scores)), 4),
        "top5_features": fi["feature"].head(5).tolist(),
    }

def train_regressor(target, df, feature_cols, params, sk, out_dir):
    y_raw = df[target].dropna()
    X_raw = df.loc[y_raw.index, feature_cols]
    y = y_raw.values
    
    pbar = tqdm(total=100, desc=f"  Training {target}", leave=False, position=0)
    pbar.update(10)
    
    sample_w = build_sample_weights(X_raw, FEATURE_WEIGHTS)
    X_tr, X_te, y_tr, y_te, sw_tr, _ = sk["tts"](
        X_raw, y, sample_w, test_size=0.2, random_state=42
    )
    
    pbar.update(20)
    
    model = sk["RFR"](**params["reg"])
    model.fit(X_tr, y_tr, sample_weight=sw_tr)
    pbar.update(40)
    
    y_pred = model.predict(X_te)
    mae = sk["mae"](y_te, y_pred)
    rmse = float(np.sqrt(sk["mse"](y_te, y_pred)))
    r2 = sk["r2"](y_te, y_pred)
    pbar.update(10)
    
    kf = sk["KF"](n_splits=5, shuffle=True, random_state=42)
    cv_scores = sk["cvs"](model, X_raw, y, cv=kf, scoring="r2", n_jobs=-1)
    pbar.update(10)
    
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    model_path = out_dir / f"{target}_rf_regressor.pkl"
    sk["jl"].dump({"model": model, "features": feature_cols}, model_path)
    fi.to_csv(out_dir / f"fi_{target}.csv", index=False)
    
    pbar.update(10)
    pbar.close()
    
    return {
        "target": target, "type": "regression",
        "n_train": len(X_tr), "n_test": len(X_te),
        "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4),
        "cv_r2_mean": round(float(np.mean(cv_scores)), 4),
        "cv_r2_std": round(float(np.std(cv_scores)), 4),
        "top5_features": fi["feature"].head(5).tolist(),
    }

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--targets", nargs="+", help="Targets to train")
    parser.add_argument("--checkpoint", help="Checkpoint file")
    args = parser.parse_args()
    
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("  INTERACTIVE TRAINING - With Progress Bars")
    print("="*60)
    print(f"\n  Dataset: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}")
    
    all_label_cols = list(TARGETS.keys())
    feature_cols = [c for c in df.columns if c not in all_label_cols]
    print(f"  Features: {len(feature_cols)}")
    print(f"  Mode: FULL (500 trees, 5-fold CV)")
    
    # Check checkpoint
    checkpoint = {}
    if args.checkpoint and Path(args.checkpoint).exists():
        with open(args.checkpoint, 'r') as f:
            checkpoint = json.load(f)
        print(f"  Resuming from checkpoint: {len(checkpoint.get('completed', []))} models done")
    
    sk = load_sklearn()
    all_metrics = []
    completed = checkpoint.get('completed', [])
    
    # Filter targets to train
    targets_to_train = args.targets if args.targets else list(TARGETS.keys())
    
    for target in targets_to_train:
        if target in completed:
            print(f"\n✅ SKIPPING {target} - already trained")
            continue
        
        if target not in TARGETS:
            print(f"\n❌ Unknown target: {target}")
            continue
        
        info = TARGETS[target]
        print(f"\n{'='*60}")
        print(f"  {target}")
        print(f"  Type: {info['type'].upper()}")
        print(f"{'='*60}")
        
        if target not in df.columns:
            print("  SKIP: Column not found")
            continue
        
        valid = df[target].dropna()
        if len(valid) < 50:
            print(f"  SKIP: Only {len(valid)} samples")
            continue
        
        print(f"  Samples: {len(valid):,}")
        start = time.time()
        
        try:
            if info["type"] == "classification":
                m = train_classifier(target, df, feature_cols, FULL_PARAMS, sk, out_dir)
            else:
                m = train_regressor(target, df, feature_cols, FULL_PARAMS, sk, out_dir)
            
            elapsed = time.time() - start
            all_metrics.append(m)
            completed.append(target)
            
            if m["type"] == "classification":
                print(f"\n  ✅ Accuracy: {m['accuracy']:.3f}")
                print(f"     F1 Macro: {m['f1_macro']:.3f}")
                print(f"     CV F1: {m['cv_f1_mean']:.3f} ± {m['cv_f1_std']:.3f}")
            else:
                print(f"\n  ✅ R²: {m['r2']:.3f}")
                print(f"     CV R²: {m['cv_r2_mean']:.3f} ± {m['cv_r2_std']:.3f}")
                print(f"     MAE: {m['mae']:.4f}")
            
            print(f"     Time: {elapsed/60:.1f} minutes")
            print(f"     Top Features: {', '.join(m['top5_features'][:3])}")
            
            # Save checkpoint
            if args.checkpoint:
                with open(args.checkpoint, 'w') as f:
                    json.dump({"completed": completed}, f)
                print(f"  💾 Checkpoint saved")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        gc.collect()
    
    # Save final report
    if all_metrics:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = ["="*60, f"  Training Complete - {timestamp}", "="*60]
        lines.append(f"\n  Models trained: {len(all_metrics)}")
        lines.append("\n  Classification Results:")
        for m in all_metrics:
            if m["type"] == "classification":
                lines.append(f"    {m['target']:<42} Acc: {m['accuracy']:.4f}  F1: {m['f1_macro']:.4f}")
        lines.append("\n  Regression Results:")
        for m in all_metrics:
            if m["type"] == "regression":
                lines.append(f"    {m['target']:<42} R²: {m['r2']:.4f}  MAE: {m['mae']:.4f}")
        
        (out_dir / "training_report.txt").write_text("\n".join(lines))
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print(f"  Models: {out_dir}/")

if __name__ == "__main__":
    main()
