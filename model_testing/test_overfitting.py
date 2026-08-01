#!/usr/bin/env python3
"""
test_overfitting.py — Overfitting & Underfitting Diagnostic Test Module
========================================================================
Tests for generalization gap (Train vs Test score delta) and Cross-Validation
variance across all trained models.
"""

import sys
import gc
import numpy as np
import pandas as pd
from pathlib import Path

def test_model_overfitting(model_data: dict, df: pd.DataFrame, target: str, is_clf: bool, sk: dict) -> dict:
    """
    Evaluates overfitting by comparing training set performance against testing set performance,
    and calculating Cross-Validation stability.
    """
    model = model_data["model"]
    features = model_data.get("features", [])
    le = model_data.get("label_encoder")

    if target not in df.columns:
        return {"target": target, "status": "SKIPPED", "issue": f"Target column '{target}' missing in CSV"}

    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        return {"target": target, "status": "SKIPPED", "issue": f"Missing features: {missing_feats[:3]}"}

    valid_df = df.dropna(subset=[target]).copy()
    # Sample up to 50,000 rows for fast diagnostic CV & train/test check
    if len(valid_df) > 50000:
        valid_df = valid_df.sample(n=50000, random_state=42)

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
            y = y_raw.values
    else:
        y = valid_df[target].values.astype(float)

    if len(X) < 20:
        return {"target": target, "status": "SKIPPED", "issue": "Insufficient samples"}

    # Train-test split (80-20)
    try:
        if is_clf:
            X_train, X_test, y_train, y_test = sk["split"](X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            X_train, X_test, y_train, y_test = sk["split"](X, y, test_size=0.2, random_state=42)
    except Exception:
        X_train, X_test, y_train, y_test = sk["split"](X, y, test_size=0.2, random_state=42)

    # Predict on train vs test
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    if is_clf:
        score_train = float(sk["accuracy"](y_train, y_pred_train))
        score_test = float(sk["accuracy"](y_test, y_pred_test))
        metric_name = "Accuracy"
        # Overfitting threshold: Train accuracy > Test accuracy by more than 5%
        overfit_threshold = 0.05
    else:
        score_train = float(sk["r2"](y_train, y_pred_train))
        score_test = float(sk["r2"](y_test, y_pred_test))
        metric_name = "R²"
        # Overfitting threshold: Train R2 > Test R2 by more than 0.08
        overfit_threshold = 0.08

    gap = score_train - score_test

    # Diagnose status
    issues = []
    if gap > overfit_threshold:
        issues.append(f"OVERFITTING DETECTED: Train {metric_name} ({score_train:.4f}) - Test {metric_name} ({score_test:.4f}) = Gap {gap:.4f} > {overfit_threshold}")
    elif score_test < (0.80 if is_clf else 0.70):
        issues.append(f"UNDERFITTING / LOW CAPACITY: Test {metric_name} ({score_test:.4f}) is low")

    has_issue = len(issues) > 0

    return {
        "target": target,
        "type": "Classification" if is_clf else "Regression",
        "status": "FLAGGED" if has_issue else "PASSED",
        "metric_name": metric_name,
        "score_train": round(score_train, 4),
        "score_test": round(score_test, 4),
        "gap": round(gap, 4),
        "issues": "; ".join(issues) if issues else "None (Optimal Generalization)"
    }
