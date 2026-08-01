#!/usr/bin/env python3
"""
test_data_leakage.py — Data Leakage & Contamination Diagnostic Test Module
===========================================================================
Tests for feature-target correlation leakage (|r| > 0.999) and exact duplicate
sample leakage between train and test datasets.
"""

import numpy as np
import pandas as pd

def test_model_data_leakage(model_data: dict, df: pd.DataFrame, target: str, is_clf: bool, sk: dict) -> dict:
    """
    Checks for target leakage (features excessively correlated with target)
    and train/test data contamination (exact duplicate rows).
    """
    features = model_data.get("features", [])
    le = model_data.get("label_encoder")

    if target not in df.columns:
        return {"target": target, "status": "SKIPPED", "issue": f"Target column '{target}' missing"}

    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        return {"target": target, "status": "SKIPPED", "issue": f"Missing features: {missing_feats[:3]}"}

    valid_df = df.dropna(subset=[target]).copy()
    if len(valid_df) > 30000:
        valid_df = valid_df.sample(n=30000, random_state=42)

    X = valid_df[features]
    if is_clf:
        y_raw = valid_df[target].astype(str)
        if le is not None:
            try:
                y = le.transform(y_raw)
            except ValueError:
                y = np.array([le.transform([v])[0] if v in le.classes_ else -1 for v in y_raw])
                mask = y != -1
                X, y = X[mask], y[mask]
        else:
            y = pd.Series(y_raw).astype("category").cat.codes.values
    else:
        y = valid_df[target].values.astype(float)

    issues = []

    # 1. Feature-Target Correlation Check (Leakage Detection)
    high_corr_features = []
    for col in features:
        if col == target:
            issues.append(f"CRITICAL LEAKAGE: Target column '{target}' included in model feature list")
            continue
        feat_vals = pd.to_numeric(X[col], errors="coerce").values
        valid_mask = ~np.isnan(feat_vals) & ~np.isnan(y)
        if valid_mask.sum() > 10:
            corr = np.corrcoef(feat_vals[valid_mask], y[valid_mask])[0, 1]
            if not np.isnan(corr) and abs(corr) >= 0.999:
                high_corr_features.append(f"{col} (r={corr:.4f})")

    if high_corr_features:
        issues.append(f"TARGET LEAKAGE SUSPECTED: Features near-identically correlated with target: {', '.join(high_corr_features[:3])}")

    # 2. Train-Test Sample Contamination Check (Duplicate rows)
    X_train, X_test = sk["split"](X, test_size=0.2, random_state=42)
    train_tuples = set(tuple(x) for x in X_train.values[:5000])
    test_tuples = set(tuple(x) for x in X_test.values[:1000])
    overlap_count = len(train_tuples.intersection(test_tuples))
    overlap_pct = (overlap_count / max(1, len(test_tuples))) * 100.0

    if overlap_pct > 5.0:
        issues.append(f"DATA CONTAMINATION: {overlap_count} identical samples ({overlap_pct:.1f}%) found in both train & test splits")

    has_issue = len(issues) > 0

    return {
        "target": target,
        "type": "Classification" if is_clf else "Regression",
        "status": "FLAGGED" if has_issue else "PASSED",
        "leakage_features_count": len(high_corr_features),
        "duplicate_overlap_pct": round(overlap_pct, 2),
        "issues": "; ".join(issues) if issues else "Clean (No Leakage or Contamination)"
    }
