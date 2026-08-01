#!/usr/bin/env python3
"""
test_class_imbalance.py — Class Imbalance & Minority Class Collapse Test Module
=================================================================================
Tests classification models for minority class collapse, zero-precision/recall classes,
and verifies model superiority over a Dummy Classifier (majority class strategy).
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

def test_model_class_imbalance(model_data: dict, df: pd.DataFrame, target: str, is_clf: bool, sk: dict) -> dict:
    """
    Evaluates classification models to ensure severe class imbalance hasn't caused
    the model to ignore minority classes or perform worse than trivial guessing.
    """
    if not is_clf:
        return {"target": target, "status": "N/A", "issues": "N/A (Regression Target)"}

    model = model_data["model"]
    features = model_data.get("features", [])
    le = model_data.get("label_encoder")

    if target not in df.columns:
        return {"target": target, "status": "SKIPPED", "issues": f"Target '{target}' missing"}

    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        return {"target": target, "status": "SKIPPED", "issues": f"Missing features: {missing_feats[:3]}"}

    valid_df = df.dropna(subset=[target]).copy()
    if len(valid_df) > 30000:
        valid_df = valid_df.sample(n=30000, random_state=42)

    X = valid_df[features]
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

    try:
        X_train, X_test, y_train, y_test = sk["split"](X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = sk["split"](X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    model_acc = float(sk["accuracy"](y_test, y_pred))

    # Evaluate Dummy Classifier baseline (always predict majority class)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_acc = float(sk["accuracy"](y_test, dummy_pred))

    acc_gain = model_acc - dummy_acc

    # Per-class recall / precision check
    recalls = sk["recall"](y_test, y_pred, average=None, zero_division=0)
    precisions = sk["precision"](y_test, y_pred, average=None, zero_division=0)

    collapsed_classes = []
    if le is not None:
        classes = list(le.classes_)
    else:
        classes = [str(c) for c in np.unique(y_test)]

    for idx, cls_name in enumerate(classes[:len(recalls)]):
        if recalls[idx] == 0.0 or precisions[idx] == 0.0:
            collapsed_classes.append(str(cls_name))

    issues = []
    if collapsed_classes:
        issues.append(f"MINORITY CLASS COLLAPSE: Classes with 0% recall/precision: {', '.join(collapsed_classes)}")

    if acc_gain <= 0.01 and len(set(y_test)) > 1:
        issues.append(f"NO LIFT OVER MAJORTY BASELINE: Model accuracy ({model_acc*100:.1f}%) offers no significant improvement over Dummy Baseline ({dummy_acc*100:.1f}%)")

    has_issue = len(issues) > 0

    return {
        "target": target,
        "type": "Classification",
        "status": "FLAGGED" if has_issue else "PASSED",
        "model_accuracy": round(model_acc, 4),
        "dummy_baseline_acc": round(dummy_acc, 4),
        "lift_over_dummy": round(acc_gain, 4),
        "collapsed_classes_count": len(collapsed_classes),
        "issues": "; ".join(issues) if issues else "Balanced (All Classes Predicted Accurately)"
    }
