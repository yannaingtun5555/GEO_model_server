#!/usr/bin/env python3
"""
train.py — Professional ML Training Pipeline
==============================================
Trains and evaluates models for all 17 predictions defined in prediction_req.md.

Architecture
────────────
  Predictions  1-11 : CropSuitabilityClassifier  (4-class: excellent/good/moderate/poor)
  Prediction   12   : CropHealthScoreRegressor    (0.0 – 1.0)
  Prediction   13   : CropYieldRegressor          (tons/ha)
  Prediction   14   : IrrigationNeedClassifier    (0 / 1 / 2)
  Predictions  15-17: CurrentMonthRegressor       (precip / temp / solar)

Models used: GradientBoostingClassifier / Regressor (primary),
             RandomForestClassifier / Regressor (secondary for comparison).
Both are ensemble tree methods that handle mixed numeric features well,
require no feature scaling, and naturally support feature importance.

Feature Weighting Strategy
───────────────────────────
sklearn tree models use feature_importances_ internally, but we guide them
by providing sample weights and carefully chosen hyperparameters.
We also expose explicit "domain weight" multipliers that are used to
synthetically repeat high-importance features, giving the model
stronger signal on agronomically critical inputs.

Usage
─────
    python scripts/train.py --data data/combined/combined_dataset.csv
    python scripts/train.py --data data/combined/combined_dataset.csv --quick
    python scripts/train.py --data data/combined/combined_dataset.csv --target crop_yield_t_ha
    python scripts/train.py --help

Output
──────
    models/
        <target>_<model_type>.pkl   — trained model artifact
        feature_importance_<target>.csv
        metrics_summary.csv
        training_report.txt

Requirements (install on training machine)
──────────────────────────────────────────
    pip install scikit-learn pandas numpy joblib tabulate
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE    = PROJECT_ROOT / "data" / "combined" / "combined_dataset.csv"
MODELS_DIR   = PROJECT_ROOT / "models"

# ─────────────────────────────────────────────────────────────────────────────
# Domain-expert feature weights (1.0 = normal importance, >1 = boost signal)
# These are used to synthetically upweight features via sample repetition
# and also serve as documentation of agronomic importance.
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_WEIGHTS = {
    # ── Soil fertility (critical for yield and health) ──────────────────────
    "soil_soc_g_kg_0_30cm":              3.0,  # Organic carbon = top fertility indicator
    "soil_cec_cmol_kg_0_30cm":           2.5,  # Nutrient holding capacity
    "soil_ph_h2o_0_30cm":                2.5,  # pH governs nutrient availability
    "soil_clay_pct_0_30cm":              1.5,
    "soil_sand_pct_0_30cm":              1.5,
    "soil_silt_pct_0_30cm":              1.2,

    # ── Water availability ───────────────────────────────────────────────────
    "era5_soil_moisture_m3_m3_mean":     3.0,  # Most direct water stress indicator
    "era5_soil_moisture_m3_m3_min":      2.5,  # Drought stress severity
    "era5_soil_moisture_m3_m3_cv":       1.8,  # Moisture predictability
    "ndwi_mcf_median_mean":              2.0,  # Satellite water index
    "ndwi_mcf_median_max":               1.5,
    "distance_to_surface_water_m":       1.8,
    "surface_water_occurrence_pct":      1.5,
    "surface_water_seasonality_months":  1.3,

    # ── Precipitation ────────────────────────────────────────────────────────
    "chirps_precipitation_mm_mean":      2.5,  # Annual rainfall average
    "chirps_precipitation_mm_min":       2.0,  # Dry-season minimum
    "chirps_precipitation_mm_cv":        1.8,  # Rainfall reliability
    "chirps_precipitation_mm_max":       1.5,
    "chirps_precipitation_mm_range":     1.3,
    "chirps_precipitation_mm":           1.5,  # Current month

    # ── Vegetation health ────────────────────────────────────────────────────
    "ndvi_median_mean":                  2.5,  # Overall vegetation vigor
    "ndvi_median_growing_season_mean":   2.5,  # Growing-season health
    "ndvi_median_max":                   2.0,  # Peak crop growth
    "ndvi_median_min":                   1.5,

    # ── Temperature ─────────────────────────────────────────────────────────
    "mean_temperature_c_mean":           2.0,
    "mean_temperature_c_max":            1.8,  # Heat stress
    "mean_temperature_c_min":            1.8,  # Chilling/frost risk
    "mean_temperature_c_range":          1.5,  # Seasonality
    "mean_temperature_c":                1.5,  # Current month

    # ── Solar radiation ──────────────────────────────────────────────────────
    "solar_radiation_mj_m2_day_mean":    1.8,
    "solar_radiation_mj_m2_day_max":     1.5,
    "solar_radiation_mj_m2_day":         1.5,

    # ── Terrain ─────────────────────────────────────────────────────────────
    "elevation_m":                       1.8,
    "slope_degrees":                     1.5,
    "aspect_degrees":                    1.0,

    # ── SAR backscatter ──────────────────────────────────────────────────────
    "s1_vh_db_median_mean":              1.3,
    "s1_vv_db_median_mean":              1.3,

    # ── Temporal / regional context ─────────────────────────────────────────
    "data_month":                        1.5,  # Seasonality
    "region_ayeyawaddy":                 1.0,
    "region_bago":                       1.0,
    "region_magway":                     1.0,
    "region_mandalay":                   1.0,
    "region_sagaing":                    1.0,
    "region_yangon":                     1.0,

    # ── Regional crop planting percentages (Rice dominance & crop share) ───────
    "crop_area_pct_monsoon_rice":        3.5,  # Rice dominance primary influence
    "crop_area_pct_dry_season_rice":     3.0,  # Dry-season rice share
    "crop_area_pct_sesame":              2.5,
    "crop_area_pct_groundnut":           2.5,
    "crop_area_pct_black_gram":          2.5,
    "crop_area_pct_green_gram":          2.5,
    "crop_area_pct_pigeon_pea":          2.5,
    "crop_area_pct_maize":               2.0,
    "crop_area_pct_sugarcane":           2.0,
    "crop_area_pct_cassava":             2.0,
    "crop_area_pct_durian":              2.0,
    "crop_area_pct_mangosteen":          2.0,
    "crop_area_pct_longan":              2.0,
    "crop_area_pct_mango":               2.0,
    "crop_area_pct_chili":               2.0,
    "crop_area_pct_tomato":              2.0,
    "crop_area_pct_rubber":              2.0,

    # ── Infrastructure & Landcover features ──────────────────────────────────
    "distance_to_road_km":               2.0,
    "road_density_km_per_sqkm":          1.8,
    "distance_to_railway_km":            1.5,
    "railway_density_km_per_sqkm":       1.2,
    "distance_to_river_km":              2.2,
    "river_density_km_per_sqkm":         1.8,
    "urban_fraction":                    1.5,
    "builtup_fraction":                  1.5,
    "cropland_fraction":                 2.5,
    "non_cropland_fraction":             1.5,
    "permanent_water_fraction":          1.8,
    "population_density":                1.5,
    "valid_agriculture_mask":            2.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Prediction targets definition
# ─────────────────────────────────────────────────────────────────────────────
SUITABILITY_LABEL_ORDER = ["poor", "moderate", "good", "excellent"]

TARGETS = {
    # ── Crop suitability (17 classifiers) ────────────────────────────────────
    "crop_suitability_monsoon_rice":    {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_dry_season_rice": {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_maize":           {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_sugarcane":       {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_cassava":         {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_durian":          {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_mangosteen":      {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_longan":          {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_mango":           {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_chili":           {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_tomato":          {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_black_gram":      {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_green_gram":      {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_pigeon_pea":      {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_groundnut":       {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_sesame":          {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    "crop_suitability_rubber":          {"type": "classification", "classes": SUITABILITY_LABEL_ORDER},
    # ── Health score (regressor) ──────────────────────────────────────────────
    "crop_health_score":                {"type": "regression"},
    # ── Yield (regressor) ────────────────────────────────────────────────────
    "crop_yield_t_ha":                  {"type": "regression"},
    # ── Irrigation need (classifier) ─────────────────────────────────────────
    "irrigation_need":                  {"type": "classification", "classes": [0, 1, 2]},
    # ── Current-month pass-throughs (regressors) ──────────────────────────────
    "current_month_precipitation_mm":       {"type": "regression"},
    "current_month_mean_temperature_c":     {"type": "regression"},
    "current_month_solar_rad_mj_m2_day":    {"type": "regression"},
    # ── Climate Risk & Farm Management (7 targets) ───────────────────────────
    "flood_risk_level":                 {"type": "classification", "classes": [0, 1, 2]},
    "drought_risk_score":               {"type": "regression"},
    "heat_stress_risk":                 {"type": "classification", "classes": [0, 1, 2]},
    "optimal_planting_month":           {"type": "classification", "classes": list(range(1, 13))},
    "nitrogen_requirement_level":       {"type": "classification", "classes": [0, 1, 2]},
    "phosphorus_requirement_level":     {"type": "classification", "classes": [0, 1, 2]},
    "soil_erosion_risk":                {"type": "classification", "classes": [0, 1, 2]},
    # ── Market, Supply Chain, Urbanization, Water & GDP Models (10 targets) ──
    "market_integration_score":          {"type": "regression"},
    "post_harvest_loss_risk":            {"type": "regression"},
    "supply_chain_efficiency":           {"type": "regression"},
    "cold_chain_potential":              {"type": "regression"},
    "agricultural_land_conversion_risk": {"type": "regression"},
    "urban_encroachment_risk":           {"type": "regression"},
    "irrigation_potential":              {"type": "regression"},
    "surface_water_occurrence":          {"type": "regression"},
    "water_scarcity_risk":               {"type": "regression"},
    "agricultural_gdp_forecast":         {"type": "regression"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Model hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
QUICK_PARAMS = {
    "clf": dict(n_estimators=100, max_depth=8,  min_samples_leaf=5,
                random_state=42, n_jobs=-1),
    "reg": dict(n_estimators=100, max_depth=8,  min_samples_leaf=5,
                random_state=42, n_jobs=-1),
}

FULL_PARAMS = {
    "clf": dict(n_estimators=500, max_depth=12, min_samples_leaf=3,
                max_features="sqrt", random_state=42, n_jobs=-1,
                class_weight="balanced"),   # handles class imbalance
    "reg": dict(n_estimators=500, max_depth=12, min_samples_leaf=3,
                max_features="sqrt", random_state=42, n_jobs=-1),
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_sklearn():
    """Lazy import sklearn and optional gradient boosting libraries."""
    try:
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor,
            ExtraTreesClassifier, ExtraTreesRegressor,
            HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        )
        from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
        from sklearn.metrics import (
            accuracy_score, f1_score, classification_report,
            mean_absolute_error, mean_squared_error, r2_score,
            confusion_matrix
        )
        from sklearn.preprocessing import LabelEncoder
        import joblib

        lgb_clf, lgb_reg = None, None
        try:
            import lightgbm as lgb
            lgb_clf = lgb.LGBMClassifier
            lgb_reg = lgb.LGBMRegressor
        except ImportError:
            pass

        xgb_clf, xgb_reg = None, None
        try:
            import xgboost as xgb
            xgb_clf = xgb.XGBClassifier
            xgb_reg = xgb.XGBRegressor
        except ImportError:
            pass

        return {
            "RFC":     RandomForestClassifier,
            "RFR":     RandomForestRegressor,
            "ETC":     ExtraTreesClassifier,
            "ETR":     ExtraTreesRegressor,
            "HGBC":    HistGradientBoostingClassifier,
            "HGBR":    HistGradientBoostingRegressor,
            "LGBM_C":  lgb_clf,
            "LGBM_R":  lgb_reg,
            "XGB_C":   xgb_clf,
            "XGB_R":   xgb_reg,
            "tts":     train_test_split,
            "SKF":     StratifiedKFold,
            "KF":      KFold,
            "cvs":     cross_val_score,
            "acc":     accuracy_score,
            "f1":      f1_score,
            "crpt":    classification_report,
            "mae":     mean_absolute_error,
            "mse":     mean_squared_error,
            "r2":      r2_score,
            "cm":      confusion_matrix,
            "LE":      LabelEncoder,
            "jl":      joblib,
        }
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Install with:  pip install scikit-learn pandas numpy joblib")
        sys.exit(1)


def build_sample_weights(X: pd.DataFrame, feature_weights: dict) -> np.ndarray:
    """
    Build per-sample weights that reflect domain feature importance.
    Strategy: weighted average of top features normalised per-sample.
    Rows with high values in high-weight features get higher sample weight,
    helping the model focus on agronomically relevant signal.
    """
    w = np.ones(len(X))
    total_weight = 0.0
    for col, fw in feature_weights.items():
        if col in X.columns and fw > 1.0:
            series = pd.to_numeric(X[col], errors="coerce").fillna(0)
            # Normalize to 0-1 then scale by domain weight
            rng = series.max() - series.min()
            if rng > 0:
                norm = (series - series.min()) / rng
                w = w + norm.values * (fw - 1.0)
                total_weight += (fw - 1.0)
    # Normalize to mean=1.0
    w = w / w.mean()
    return w.clip(0.5, 3.0)   # cap to avoid extreme weights


def get_feature_cols(df: pd.DataFrame, label_cols: list) -> list:
    """Return feature columns = all columns except labels."""
    return [c for c in df.columns if c not in label_cols]


def print_banner(text: str):
    line = "═" * 65
    print(f"\n{line}\n  {text}\n{line}")


def print_section(text: str):
    print(f"\n── {text} {'─'*(60-len(text))}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-target training and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(target: str, df: pd.DataFrame, feature_cols: list,
                     params: dict, sk: dict, out_dir: Path, quick: bool) -> dict:
    """Train and evaluate a classification model."""
    info     = TARGETS[target]
    y_raw    = df[target].dropna()
    X_raw    = df.loc[y_raw.index, feature_cols]

    # Encode string labels to integers
    le = sk["LE"]()
    if info.get("classes") and isinstance(info["classes"][0], str):
        le.fit(info["classes"])
    else:
        le.fit(sorted(y_raw.unique()))
    y = le.transform(y_raw.astype(str))

    sample_w = build_sample_weights(X_raw, FEATURE_WEIGHTS)
    # Stratified split; fall back to random if any class has < 2 samples
    try:
        X_tr, X_te, y_tr, y_te, sw_tr, _ = sk["tts"](
            X_raw, y, sample_w, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te, sw_tr, _ = sk["tts"](
            X_raw, y, sample_w, test_size=0.2, random_state=42
        )

    p = params["clf"].copy()
    p.pop("class_weight", None)   # RF supports class_weight differently
    model = sk["RFC"](**p, class_weight="balanced")
    model.fit(X_tr, y_tr, sample_weight=sw_tr)

    # Evaluation
    y_pred  = model.predict(X_te)
    acc     = sk["acc"](y_te, y_pred)
    f1_mac  = sk["f1"](y_te, y_pred, average="macro",  zero_division=0)
    f1_wt   = sk["f1"](y_te, y_pred, average="weighted", zero_division=0)
    classes_str = le.inverse_transform(sorted(set(y)))

    report = sk["crpt"](y_te, y_pred,
                        target_names=[str(c) for c in classes_str],
                        zero_division=0)

    # Cross-validation (skip in quick mode for speed)
    cv_scores = []
    if not quick:
        kf = sk["SKF"](n_splits=5, shuffle=True, random_state=42)
        cv_scores = sk["cvs"](model, X_raw, y, cv=kf, scoring="f1_macro", n_jobs=-1)

    # Feature importance
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Save
    model_path = out_dir / f"{target}_rf_classifier.pkl"
    sk["jl"].dump({"model": model, "label_encoder": le, "features": feature_cols}, model_path)
    fi.to_csv(out_dir / f"fi_{target}.csv", index=False)

    metrics = {
        "target":       target,
        "type":         "classification",
        "n_train":      len(X_tr),
        "n_test":       len(X_te),
        "accuracy":     round(acc, 4),
        "f1_macro":     round(f1_mac, 4),
        "f1_weighted":  round(f1_wt, 4),
        "cv_f1_macro_mean": round(float(np.mean(cv_scores)), 4) if len(cv_scores) > 0 else None,
        "cv_f1_macro_std":  round(float(np.std(cv_scores)),  4) if len(cv_scores) > 0 else None,
        "top5_features": fi["feature"].head(5).tolist(),
        "report":        report,
    }
    return metrics


def train_regressor(target: str, df: pd.DataFrame, feature_cols: list,
                    params: dict, sk: dict, out_dir: Path, quick: bool) -> dict:
    """Train and evaluate a regression model."""
    y_raw    = df[target].dropna()
    X_raw    = df.loc[y_raw.index, feature_cols]
    y        = y_raw.values

    sample_w = build_sample_weights(X_raw, FEATURE_WEIGHTS)
    X_tr, X_te, y_tr, y_te, sw_tr, _ = sk["tts"](
        X_raw, y, sample_w, test_size=0.2, random_state=42
    )

    model = sk["RFR"](**params["reg"])
    model.fit(X_tr, y_tr, sample_weight=sw_tr)

    y_pred = model.predict(X_te)
    mae    = sk["mae"](y_te, y_pred)
    rmse   = float(np.sqrt(sk["mse"](y_te, y_pred)))
    r2     = sk["r2"](y_te, y_pred)

    # Cross-validation
    cv_scores = []
    if not quick:
        kf = sk["KF"](n_splits=5, shuffle=True, random_state=42)
        cv_scores = sk["cvs"](model, X_raw, y, cv=kf, scoring="r2", n_jobs=-1)

    # Residuals analysis
    residuals = y_te - y_pred
    res_mean  = float(np.mean(residuals))
    res_std   = float(np.std(residuals))

    # Feature importance
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Save
    model_path = out_dir / f"{target}_rf_regressor.pkl"
    sk["jl"].dump({"model": model, "features": feature_cols}, model_path)
    fi.to_csv(out_dir / f"fi_{target}.csv", index=False)

    metrics = {
        "target":           target,
        "type":             "regression",
        "n_train":          len(X_tr),
        "n_test":           len(X_te),
        "mae":              round(mae,  4),
        "rmse":             round(rmse, 4),
        "r2":               round(r2,   4),
        "cv_r2_mean":       round(float(np.mean(cv_scores)), 4) if len(cv_scores) > 0 else None,
        "cv_r2_std":        round(float(np.std(cv_scores)),  4) if len(cv_scores) > 0 else None,
        "residual_mean":    round(res_mean, 4),
        "residual_std":     round(res_std,  4),
        "y_test_min":       round(float(y_te.min()), 4),
        "y_test_max":       round(float(y_te.max()), 4),
        "y_pred_min":       round(float(y_pred.min()), 4),
        "y_pred_max":       round(float(y_pred.max()), 4),
        "top5_features":    fi["feature"].head(5).tolist(),
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Report writing
# ─────────────────────────────────────────────────────────────────────────────

def write_report(all_metrics: list, out_dir: Path, n_rows: int, n_features: int):
    """Write a human-readable training report and metrics CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("═" * 65)
    lines.append("  Myanmar Agricultural ML — Training Report")
    lines.append(f"  Generated: {timestamp}")
    lines.append("═" * 65)
    lines.append(f"\n  Dataset: {n_rows:,} samples × {n_features} features")
    lines.append(f"  Models trained: {len(all_metrics)}\n")

    clf_rows, reg_rows = [], []

    for m in all_metrics:
        lines.append("─" * 65)
        lines.append(f"  TARGET: {m['target']}")
        lines.append(f"  Type  : {m['type'].upper()}")
        lines.append(f"  Train : {m['n_train']:,}  |  Test: {m['n_test']:,}")

        if m["type"] == "classification":
            lines.append(f"  Accuracy    : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
            lines.append(f"  F1 (macro)  : {m['f1_macro']:.4f}")
            lines.append(f"  F1 (weighted): {m['f1_weighted']:.4f}")
            if m["cv_f1_macro_mean"] is not None:
                lines.append(f"  CV F1 macro : {m['cv_f1_macro_mean']:.4f} ± {m['cv_f1_macro_std']:.4f}")
            lines.append(f"  Top features: {', '.join(m['top5_features'][:3])}")
            lines.append(f"\n{m['report']}")
            clf_rows.append({
                "target": m["target"],
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "f1_weighted": m["f1_weighted"],
                "cv_f1_macro_mean": m["cv_f1_macro_mean"],
            })
        else:
            lines.append(f"  MAE    : {m['mae']:.4f}")
            lines.append(f"  RMSE   : {m['rmse']:.4f}")
            lines.append(f"  R²     : {m['r2']:.4f}")
            if m["cv_r2_mean"] is not None:
                lines.append(f"  CV R²  : {m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}")
            lines.append(f"  Residual: mean={m['residual_mean']:.4f}  std={m['residual_std']:.4f}")
            lines.append(f"  Top features: {', '.join(m['top5_features'][:3])}")
            reg_rows.append({
                "target": m["target"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "cv_r2_mean": m["cv_r2_mean"],
            })

    lines.append("\n" + "═" * 65)
    lines.append("  SUMMARY")
    lines.append("═" * 65)

    if clf_rows:
        lines.append("\n  Classification models:")
        lines.append(f"  {'Target':<42} {'Accuracy':>9} {'F1 Mac':>8} {'F1 Wt':>8}")
        lines.append(f"  {'─'*42} {'─'*9} {'─'*8} {'─'*8}")
        for r in clf_rows:
            lines.append(f"  {r['target']:<42} {r['accuracy']:>9.4f} {r['f1_macro']:>8.4f} {r['f1_weighted']:>8.4f}")

    if reg_rows:
        lines.append("\n  Regression models:")
        lines.append(f"  {'Target':<42} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
        lines.append(f"  {'─'*42} {'─'*8} {'─'*8} {'─'*8}")
        for r in reg_rows:
            lines.append(f"  {r['target']:<42} {r['mae']:>8.4f} {r['rmse']:>8.4f} {r['r2']:>8.4f}")

    report_text = "\n".join(lines)
    report_path = out_dir / "training_report.txt"
    report_path.write_text(report_text)

    # Save metrics CSV
    all_flat = []
    for m in all_metrics:
        row = {k: v for k, v in m.items() if k not in ("report", "top5_features")}
        row["top5_features"] = ", ".join(m.get("top5_features", []))
        all_flat.append(row)
    pd.DataFrame(all_flat).to_csv(out_dir / "metrics_summary.csv", index=False)

    return report_text


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train ML models for all 17 Myanmar Agricultural predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", default=str(DATA_FILE),
                        help=f"Path to combined_dataset.csv (default: {DATA_FILE})")
    parser.add_argument("--target", default=None,
                        help="Train only one target (e.g. crop_yield_t_ha). Default: all.")
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller model (100 trees, no CV) for fast testing.")
    parser.add_argument("--out-dir", default=str(MODELS_DIR),
                        help=f"Output directory for models/reports (default: {MODELS_DIR})")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir   = Path(args.out_dir)

    if not data_path.is_file():
        print(f"\n[ERROR] Dataset not found: {data_path}")
        print("Run:  python scripts/combine.py  first.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load sklearn
    sk = _load_sklearn()

    # ── Load data
    print_banner("Myanmar Agricultural ML — Training Pipeline")
    print(f"\n  Loading dataset: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}")

    all_label_cols = list(TARGETS.keys())
    feature_cols   = [c for c in df.columns if c not in all_label_cols]
    params         = QUICK_PARAMS if args.quick else FULL_PARAMS

    print(f"  Features: {len(feature_cols)}  |  Targets: {len(TARGETS)}")
    print(f"  Mode: {'QUICK (100 trees, no CV)' if args.quick else 'FULL (500 trees, 5-fold CV)'}")
    print(f"  Output: {out_dir}\n")

    # ── Feature weight info
    print_section("Domain Feature Weights")
    top_weights = sorted(FEATURE_WEIGHTS.items(), key=lambda x: -x[1])[:10]
    print(f"  {'Feature':<45} Weight")
    print(f"  {'─'*45} ──────")
    for feat, wt in top_weights:
        bar = "█" * int(wt * 5)
        print(f"  {feat:<45} {wt:.1f}  {bar}")

    # ── Select targets to train
    targets_to_train = {args.target: TARGETS[args.target]} if args.target else TARGETS

    if args.target and args.target not in TARGETS:
        print(f"\n[ERROR] Unknown target: {args.target}")
        print(f"Valid targets: {list(TARGETS.keys())}")
        sys.exit(1)

    # ── Training loop
    all_metrics = []
    total = len(targets_to_train)

    for i, (target, info) in enumerate(targets_to_train.items(), 1):
        print_section(f"[{i}/{total}] {target}")

        # Check if target exists and has enough data
        if target not in df.columns:
            print(f"  [SKIP] Column not found in dataset.")
            continue
        valid = df[target].dropna()
        if len(valid) < 50:
            print(f"  [SKIP] Only {len(valid)} non-null rows — not enough to train.")
            continue

        print(f"  Non-null samples: {len(valid):,}")
        if info["type"] == "classification":
            vc = df[target].value_counts()
            print(f"  Class distribution: {dict(vc)}")
        else:
            s = pd.to_numeric(df[target], errors="coerce").dropna()
            print(f"  Range: {s.min():.3f} – {s.max():.3f}  |  Mean: {s.mean():.3f}")

        print(f"  Training ...", end=" ", flush=True)
        try:
            if info["type"] == "classification":
                m = train_classifier(target, df, feature_cols, params, sk, out_dir, args.quick)
            else:
                m = train_regressor(target, df, feature_cols, params, sk, out_dir, args.quick)

            all_metrics.append(m)

            if m["type"] == "classification":
                print(f"Accuracy={m['accuracy']:.3f}  F1={m['f1_macro']:.3f}")
            else:
                print(f"R²={m['r2']:.3f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

            print(f"  Top features: {', '.join(m['top5_features'][:3])}")

        except Exception as e:
            print(f"\n  [ERROR] {e}")
            import traceback; traceback.print_exc()

    # ── Write report
    if all_metrics:
        report = write_report(all_metrics, out_dir, n_rows=len(df), n_features=len(feature_cols))
        print_banner("Training Complete")
        print(report.split("SUMMARY")[1] if "SUMMARY" in report else "")
        print(f"\n  Models saved to    : {out_dir}/")
        print(f"  Training report    : {out_dir}/training_report.txt")
        print(f"  Metrics summary    : {out_dir}/metrics_summary.csv")
        print(f"  Feature importances: {out_dir}/fi_<target>.csv")
    else:
        print("\n[WARN] No models were trained.")


if __name__ == "__main__":
    main()
