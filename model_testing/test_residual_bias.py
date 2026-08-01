#!/usr/bin/env python3
"""
test_residual_bias.py — Residual Anomalies & Prediction Bias Test Module
========================================================================
Tests regression models for systematic prediction bias (mean error != 0),
heteroscedasticity, and extreme residual outlier frequencies.
"""

import numpy as np
import pandas as pd

def test_model_residual_bias(model_data: dict, df: pd.DataFrame, target: str, is_clf: bool, sk: dict) -> dict:
    """
    Evaluates regression residuals to detect systematic bias, heteroscedasticity,
    and extreme residual outliers.
    """
    if is_clf:
        return {"target": target, "status": "N/A", "issues": "N/A (Classification Target)"}

    model = model_data["model"]
    features = model_data.get("features", [])

    if target not in df.columns:
        return {"target": target, "status": "SKIPPED", "issues": f"Target '{target}' missing"}

    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        return {"target": target, "status": "SKIPPED", "issues": f"Missing features: {missing_feats[:3]}"}

    valid_df = df.dropna(subset=[target]).copy()
    if len(valid_df) > 30000:
        valid_df = valid_df.sample(n=30000, random_state=42)

    X = valid_df[features]
    y_true = valid_df[target].values.astype(float)

    _, X_test, _, y_test = sk["split"](X, y_true, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    residuals = y_pred - y_test
    mean_bias = float(np.mean(residuals))
    rmse = float(np.sqrt(sk["mse"](y_test, y_pred)))
    target_std = float(np.std(y_test))

    issues = []

    # 1. Systematic Mean Bias Check
    # Flag if mean bias is larger than 5% of target standard deviation
    if target_std > 1e-6 and abs(mean_bias) / target_std > 0.05:
        issues.append(f"SYSTEMATIC PREDICTION BIAS: Mean error ({mean_bias:.4f}) deviates significantly from 0 (rel bias: {abs(mean_bias)/target_std*100:.1f}%)")

    # 2. Extreme Outlier Residual Check (> 3 * RMSE)
    outlier_count = int(np.sum(np.abs(residuals) > 3 * rmse))
    outlier_pct = (outlier_count / len(residuals)) * 100.0
    if outlier_pct > 1.0:
        issues.append(f"HIGH OUTLIER ERRORS: {outlier_count} test predictions ({outlier_pct:.2f}%) have errors > 3x RMSE")

    # 3. Heteroscedasticity Check (Correlation between predicted magnitude and residual magnitude)
    pred_magnitude = np.abs(y_pred)
    res_magnitude = np.abs(residuals)
    if len(pred_magnitude) > 10:
        hetero_corr = float(np.corrcoef(pred_magnitude, res_magnitude)[0, 1])
        if not np.isnan(hetero_corr) and hetero_corr > 0.40:
            issues.append(f"HETEROSCEDASTICITY DETECTED: Residual magnitude correlates with predicted scale (r={hetero_corr:.4f})")

    has_issue = len(issues) > 0

    return {
        "target": target,
        "type": "Regression",
        "status": "FLAGGED" if has_issue else "PASSED",
        "mean_bias": round(mean_bias, 4),
        "rmse": round(rmse, 4),
        "outlier_pct": round(outlier_pct, 2),
        "issues": "; ".join(issues) if issues else "Passed (Zero Bias & Homoscedastic)"
    }
