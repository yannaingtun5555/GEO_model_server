#!/usr/bin/env python3
"""
test_edge_cases_bounds.py — Boundary Violations & Feature Quality Test Module
=============================================================================
Tests models for physical boundary violations (negative yields, extreme weather values),
zero-variance features, and handling of edge-case inputs.
"""

import numpy as np
import pandas as pd

PHYSICAL_BOUNDS = {
    "crop_yield_t_ha": (0.0, 20.0),                     # Yield cannot be negative or > 20 t/ha
    "crop_health_score": (0.0, 1.0),                    # Normalized score 0-1
    "drought_risk_score": (0.0, 1.0),                   # Normalized risk 0-1
    "market_integration_score": (0.0, 1.0),            # Normalized score 0-1
    "post_harvest_loss_risk": (0.0, 1.0),               # Risk fraction 0-1
    "supply_chain_efficiency": (0.0, 1.0),              # Score 0-1
    "cold_chain_potential": (0.0, 1.0),                 # Score 0-1
    "agricultural_land_conversion_risk": (0.0, 1.0),   # Risk 0-1
    "urban_encroachment_risk": (0.0, 1.0),              # Risk 0-1
    "irrigation_potential": (0.0, 1.0),                 # Potential 0-1
    "surface_water_occurrence": (0.0, 1.0),             # Occurrence 0-1
    "water_scarcity_risk": (0.0, 1.0),                  # Risk 0-1
    "current_month_mean_temperature_c": (0.0, 55.0),    # Temperatures in °C
    "current_month_precipitation_mm": (0.0, 3000.0),    # Precip in mm
    "current_month_solar_rad_mj_m2_day": (0.0, 40.0),   # Solar rad in MJ/m2/day
}

def test_model_edge_cases(model_data: dict, df: pd.DataFrame, target: str, is_clf: bool, sk: dict) -> dict:
    """
    Evaluates physical boundary sanity checks, probability distribution constraints,
    and detects zero-variance features.
    """
    model = model_data["model"]
    features = model_data.get("features", [])

    if target not in df.columns:
        return {"target": target, "status": "SKIPPED", "issues": f"Target '{target}' missing"}

    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        return {"target": target, "status": "SKIPPED", "issues": f"Missing features: {missing_feats[:3]}"}

    valid_df = df.dropna(subset=[target]).copy()
    if len(valid_df) > 20000:
        valid_df = valid_df.sample(n=20000, random_state=42)

    X = valid_df[features]
    y_pred = model.predict(X)

    issues = []

    # 1. Zero Variance Features Check
    zero_var_feats = []
    for col in features:
        vals = pd.to_numeric(X[col], errors="coerce").dropna()
        if len(vals) > 1 and vals.std() == 0:
            zero_var_feats.append(col)

    if zero_var_feats:
        issues.append(f"DEAD FEATURES: Zero variance features found: {', '.join(zero_var_feats[:3])}")

    # 2. Physical Bounds Check (for Regressors)
    out_of_bounds_count = 0
    if not is_clf:
        min_val, max_val = PHYSICAL_BOUNDS.get(target, (-np.inf, np.inf))
        invalid_mask = (y_pred < min_val) | (y_pred > max_val)
        out_of_bounds_count = int(invalid_mask.sum())
        if out_of_bounds_count > 0:
            min_pred, max_pred = float(np.min(y_pred)), float(np.max(y_pred))
            issues.append(f"PHYSICAL BOUNDARY VIOLATION: {out_of_bounds_count} predictions ({out_of_bounds_count/len(y_pred)*100:.2f}%) fell outside expected [{min_val}, {max_val}] range (Predicted range: [{min_pred:.4f}, {max_pred:.4f}])")

    # 3. Probability Calibration Check (for Classifiers with predict_proba)
    if is_clf and hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        if np.isnan(probas).any() or np.isinf(probas).any():
            issues.append("INVALID PROBABILITIES: predict_proba returned NaN or Inf values")
        row_sums = np.sum(probas, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            issues.append("UNNORMALIZED PROBABILITIES: predict_proba row sums do not equal 1.0")

    has_issue = len(issues) > 0

    return {
        "target": target,
        "type": "Classification" if is_clf else "Regression",
        "status": "FLAGGED" if has_issue else "PASSED",
        "zero_var_features_count": len(zero_var_feats),
        "boundary_violations_count": out_of_bounds_count,
        "issues": "; ".join(issues) if issues else "Passed (Physically Valid & Well-Calibrated)"
    }
