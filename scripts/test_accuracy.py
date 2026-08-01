#!/usr/bin/env python3
"""
test_accuracy.py — Comprehensive Accuracy & Performance Testing Script
========================================================================
Loads trained models from models/ directory and evaluates their prediction accuracy,
precision, recall, F1 score, MAE, RMSE, and R2 against data in data/combined/combined_dataset.csv.

Usage:
    python scripts/test_accuracy.py
    python scripts/test_accuracy.py --data data/combined/combined_dataset.csv
    python scripts/test_accuracy.py --target crop_yield_t_ha
    python scripts/test_accuracy.py --mode full  # test on 100% of dataset
    python scripts/test_accuracy.py --mode split # test on 20% held-out test split (default)
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


def load_dependencies():
    """Import sklearn and joblib cleanly."""
    try:
        import joblib
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            classification_report, mean_absolute_error, mean_squared_error,
            r2_score, confusion_matrix
        )
        from sklearn.model_selection import train_test_split
        return {
            "joblib": joblib,
            "accuracy": accuracy_score,
            "f1": f1_score,
            "precision": precision_score,
            "recall": recall_score,
            "report": classification_report,
            "mae": mean_absolute_error,
            "mse": mean_squared_error,
            "r2": r2_score,
            "cm": confusion_matrix,
            "split": train_test_split,
        }
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run: pip install scikit-learn pandas numpy joblib tabulate")
        sys.exit(1)


def find_model_files(models_dir: Path, target_filter: str = None) -> list:
    """Find all saved .pkl model files in the models directory."""
    if not models_dir.exists():
        print(f"[ERROR] Models directory '{models_dir}' does not exist.")
        print("Please train models first using: ./run.sh train")
        sys.exit(1)

    pkl_files = list(models_dir.glob("*.pkl"))
    if target_filter:
        pkl_files = [f for f in pkl_files if target_filter in f.name]

    if not pkl_files:
        print(f"[WARNING] No .pkl model files found in '{models_dir}'.")
        sys.exit(0)

    return sorted(pkl_files)


def evaluate_classification(target: str, model_data: dict, df: pd.DataFrame,
                            mode: str, sk: dict) -> dict:
    """Evaluate a classification model."""
    model = model_data["model"]
    le = model_data.get("label_encoder")
    features = model_data.get("features")

    if target not in df.columns:
        return {"target": target, "status": "SKIPPED (Target column missing in CSV)"}

    # Ensure required features exist
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        return {"target": target, "status": f"SKIPPED (Missing features: {missing_feats[:3]})"}

    # Drop NaNs in target
    valid_df = df.dropna(subset=[target]).copy()
    X = valid_df[features]
    y_raw = valid_df[target].astype(str)

    if le is not None:
        try:
            y_true = le.transform(y_raw)
        except ValueError:
            # Handle unseen labels gracefully if any
            y_true = np.array([le.transform([val])[0] if val in le.classes_ else -1 for val in y_raw])
            mask = y_true != -1
            X = X[mask]
            y_true = y_true[mask]
    else:
        y_true = y_raw.values

    # Train / Test split vs Full dataset mode
    if mode == "split" and len(X) >= 10:
        try:
            _, X_eval, _, y_eval = sk["split"](X, y_true, test_size=0.2, random_state=42, stratify=y_true)
        except ValueError:
            _, X_eval, _, y_eval = sk["split"](X, y_true, test_size=0.2, random_state=42)
    else:
        X_eval, y_eval = X, y_true

    y_pred = model.predict(X_eval)

    # Compute metrics
    acc = sk["accuracy"](y_eval, y_pred)
    f1_macro = sk["f1"](y_eval, y_pred, average="macro", zero_division=0)
    f1_weighted = sk["f1"](y_eval, y_pred, average="weighted", zero_division=0)
    prec_macro = sk["precision"](y_eval, y_pred, average="macro", zero_division=0)
    rec_macro = sk["recall"](y_eval, y_pred, average="macro", zero_division=0)

    # Target class labels
    if le is not None:
        labels = list(range(len(le.classes_)))
        target_names = [str(c) for c in le.classes_]
    else:
        labels = sorted(list(set(np.unique(y_eval)).union(set(np.unique(y_pred)))))
        target_names = [str(c) for c in labels]

    report = sk["report"](y_eval, y_pred, labels=labels, target_names=target_names, zero_division=0)

    return {
        "target": target,
        "type": "Classification",
        "mode": mode,
        "n_samples": len(X_eval),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "report": report,
        "status": "SUCCESS"
    }


def evaluate_regression(target: str, model_data: dict, df: pd.DataFrame,
                          mode: str, sk: dict) -> dict:
    """Evaluate a regression model."""
    model = model_data["model"]
    features = model_data.get("features")

    if target not in df.columns:
        return {"target": target, "status": "SKIPPED (Target column missing in CSV)"}

    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        return {"target": target, "status": f"SKIPPED (Missing features: {missing_feats[:3]})"}

    valid_df = df.dropna(subset=[target]).copy()
    X = valid_df[features]
    y_true = valid_df[target].values.astype(float)

    if mode == "split" and len(X) >= 10:
        _, X_eval, _, y_eval = sk["split"](X, y_true, test_size=0.2, random_state=42)
    else:
        X_eval, y_eval = X, y_true

    y_pred = model.predict(X_eval)

    mae = sk["mae"](y_eval, y_pred)
    rmse = float(np.sqrt(sk["mse"](y_eval, y_pred)))
    r2 = sk["r2"](y_eval, y_pred)
    mean_bias = float(np.mean(y_pred - y_eval))

    return {
        "target": target,
        "type": "Regression",
        "mode": mode,
        "n_samples": len(X_eval),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_bias": mean_bias,
        "status": "SUCCESS"
    }


def main():
    parser = argparse.ArgumentParser(description="Test Accuracy of Trained Models on Dataset")
    parser.add_argument("--data", type=str, default=str(DATA_FILE), help="Path to combined dataset CSV")
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR), help="Path to models directory")
    parser.add_argument("--target", type=str, default=None, help="Filter to test a specific target only")
    parser.add_argument("--mode", choices=["split", "full"], default="split",
                        help="Evaluation mode: 'split' (20%% held-out test data) or 'full' (all data)")
    args = parser.parse_args()

    sk = load_dependencies()

    data_path = Path(args.data)
    models_dir = Path(args.models_dir)

    print("=====================================================================")
    print("      ACCURACY & MODEL EVALUATION REPORT FOR COMBINED DATASET        ")
    print("=====================================================================")
    print(f" Dataset Path : {data_path}")
    print(f" Models Dir   : {models_dir}")
    print(f" Eval Mode    : {args.mode.upper()} {'(20% Test Split)' if args.mode == 'split' else '(Full Dataset)'}")
    if args.target:
        print(f" Filter Target: {args.target}")
    print("---------------------------------------------------------------------\n")

    if not data_path.exists():
        print(f"[ERROR] Dataset file not found: {data_path}")
        sys.exit(1)

    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df):,} rows x {len(df.columns)} columns.\n")

    model_files = find_model_files(models_dir, args.target)
    print(f"Found {len(model_files)} model artifact(s) to evaluate.\n")

    clf_results = []
    reg_results = []

    import gc
    for idx, pkl_path in enumerate(model_files, 1):
        print(f"[{idx}/{len(model_files)}] Evaluating model: {pkl_path.name}...", flush=True)
        try:
            model_data = sk["joblib"].load(pkl_path)
        except Exception as e:
            print(f"[ERROR] Failed to load model file {pkl_path.name}: {e}", flush=True)
            continue

        # Extract target name from filename (e.g. crop_yield_t_ha_rf_regressor.pkl)
        fname = pkl_path.stem
        if fname.endswith("_rf_classifier"):
            target = fname[:-14]
            is_clf = True
        elif fname.endswith("_rf_regressor"):
            target = fname[:-13]
            is_clf = False
        else:
            target = fname
            is_clf = hasattr(model_data.get("model"), "predict_proba")

        if is_clf:
            res = evaluate_classification(target, model_data, df, args.mode, sk)
            if res["status"] == "SUCCESS":
                clf_results.append(res)
                print(f"  ✓ {target} (Classification) - Acc: {res['accuracy']*100:.2f}%, F1: {res['f1_macro']:.4f}", flush=True)
            else:
                print(f"[SKIP] {target}: {res['status']}", flush=True)
        else:
            res = evaluate_regression(target, model_data, df, args.mode, sk)
            if res["status"] == "SUCCESS":
                reg_results.append(res)
                print(f"  ✓ {target} (Regression) - R²: {res['r2']:.4f}, MAE: {res['mae']:.4f}", flush=True)
            else:
                print(f"[SKIP] {target}: {res['status']}", flush=True)

        del model_data
        gc.collect()

    # Print Classification Results
    if clf_results:
        print("\n" + "=" * 80)
        print("                       CLASSIFICATION ACCURACY                       ")
        print("=" * 80)
        clf_df = pd.DataFrame([
            {
                "Target": r["target"],
                "Samples": r["n_samples"],
                "Accuracy": f"{r['accuracy']*100:.2f}%",
                "F1 (Macro)": f"{r['f1_macro']:.4f}",
                "F1 (Weighted)": f"{r['f1_weighted']:.4f}",
                "Precision": f"{r['precision_macro']:.4f}",
                "Recall": f"{r['recall_macro']:.4f}",
            }
            for r in clf_results
        ])
        print(clf_df.to_string(index=False))

    # Print Regression Results
    if reg_results:
        print("\n" + "=" * 80)
        print("                         REGRESSION ACCURACY                         ")
        print("=" * 80)
        reg_df = pd.DataFrame([
            {
                "Target": r["target"],
                "Samples": r["n_samples"],
                "R² Score": f"{r['r2']:.4f}",
                "MAE": f"{r['mae']:.4f}",
                "RMSE": f"{r['rmse']:.4f}",
                "Mean Bias": f"{r['mean_bias']:.4f}",
            }
            for r in reg_results
        ])
        print(reg_df.to_string(index=False))

    # Save summary CSV
    out_csv = models_dir / "test_accuracy_summary.csv"
    all_summary = []
    for r in clf_results:
        all_summary.append({
            "target": r["target"],
            "type": "classification",
            "mode": r["mode"],
            "n_samples": r["n_samples"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "f1_weighted": r["f1_weighted"],
            "precision_macro": r["precision_macro"],
            "recall_macro": r["recall_macro"],
            "mae": None, "rmse": None, "r2": None, "mean_bias": None
        })
    for r in reg_results:
        all_summary.append({
            "target": r["target"],
            "type": "regression",
            "mode": r["mode"],
            "n_samples": r["n_samples"],
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "precision_macro": None, "recall_macro": None,
            "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"], "mean_bias": r["mean_bias"]
        })

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[SUMMARY] Saved accuracy report summary to: {out_csv}")

    # Detailed report file
    out_report = models_dir / "test_accuracy_report.txt"
    with out_report.open("w") as f:
        f.write("ACCURACY AND MODEL PERFORMANCE EVALUATION REPORT\n")
        f.write("================================================\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Mode: {args.mode}\n\n")
        
        if clf_results:
            f.write("CLASSIFICATION TARGETS:\n")
            f.write("----------------------\n")
            for r in clf_results:
                f.write(f"\nTarget: {r['target']}\n")
                f.write(f"Accuracy: {r['accuracy']*100:.2f}% | F1 Macro: {r['f1_macro']:.4f} | F1 Weighted: {r['f1_weighted']:.4f}\n")
                f.write("Classification Report:\n")
                f.write(r['report'])
                f.write("\n" + "-"*50 + "\n")

        if reg_results:
            f.write("\nREGRESSION TARGETS:\n")
            f.write("-------------------\n")
            for r in reg_results:
                f.write(f"\nTarget: {r['target']}\n")
                f.write(f"R²: {r['r2']:.4f} | MAE: {r['mae']:.4f} | RMSE: {r['rmse']:.4f} | Mean Bias: {r['mean_bias']:.4f}\n")
                f.write("-" * 50 + "\n")

    print(f"[REPORT] Saved detailed text report to: {out_report}\n")


if __name__ == "__main__":
    main()
