#!/usr/bin/env python3
"""
test_gp_models.py — Sequential Model Evaluation Script for `gp_models/`
========================================================================
Tests trained models from the `gp_models/` folder (or custom directory)
one by one against the dataset (`data/combined/combined_dataset.csv`).

Features:
  - Evaluates models one by one as they are trained/added to `gp_models/`
  - Automatic detection of Classifier vs Regressor
  - Supports 20% test-split mode (`--mode split`) or full dataset evaluation (`--mode full`)
  - Handles LabelEncoder decoding and missing target/feature columns gracefully
  - Displays instant metrics per model and outputs clean summary & text reports

Usage:
  python scripts/test_gp_models.py
  python scripts/test_gp_models.py --models-dir gp_models
  python scripts/test_gp_models.py --target crop_suitability_chili
  python scripts/test_gp_models.py --mode full
  ./run.sh test-gp
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_FILE  = PROJECT_ROOT / "data" / "combined" / "combined_dataset.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "gp_models"


def load_dependencies():
    """Import sklearn and joblib safely."""
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
        print(f"[ERROR] Missing required dependency: {e}")
        print("Please install requirements: pip install scikit-learn pandas numpy joblib")
        sys.exit(1)


def find_model_files(models_dir: Path, target_filter: str = None) -> list:
    """Find and return all .pkl model files sorted by name."""
    if not models_dir.exists():
        print(f"[ERROR] Models directory '{models_dir}' does not exist.")
        sys.exit(1)

    pkl_files = sorted(list(models_dir.glob("*.pkl")))
    if target_filter:
        pkl_files = [f for f in pkl_files if target_filter.lower() in f.name.lower()]

    return pkl_files


def evaluate_classification_model(target_name: str, model_data: dict, df: pd.DataFrame, mode: str, sk: dict) -> dict:
    """Evaluate a single classification model."""
    model = model_data.get("model")
    le = model_data.get("label_encoder")
    features = model_data.get("features", [])

    if target_name not in df.columns:
        return {"status": "SKIPPED", "reason": f"Target column '{target_name}' not found in CSV."}

    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        return {"status": "SKIPPED", "reason": f"Missing {len(missing_feats)} feature columns (e.g. {missing_feats[:2]})."}

    valid_df = df.dropna(subset=[target_name]).copy()
    if valid_df.empty:
        return {"status": "SKIPPED", "reason": "No non-null rows for this target column."}

    X = valid_df[features]
    y_raw = valid_df[target_name].astype(str)

    if le is not None:
        try:
            y_true = le.transform(y_raw)
        except Exception:
            y_true = np.array([le.transform([val])[0] if val in le.classes_ else -1 for val in y_raw])
            mask = y_true != -1
            X = X[mask]
            y_true = y_true[mask]
    else:
        y_true = y_raw.values

    if mode == "split" and len(X) >= 10:
        try:
            _, X_eval, _, y_eval = sk["split"](X, y_true, test_size=0.2, random_state=42, stratify=y_true)
        except ValueError:
            _, X_eval, _, y_eval = sk["split"](X, y_true, test_size=0.2, random_state=42)
    else:
        X_eval, y_eval = X, y_true

    t0 = time.time()
    y_pred = model.predict(X_eval)
    eval_latency_ms = (time.time() - t0) * 1000

    acc = sk["accuracy"](y_eval, y_pred)
    f1_macro = sk["f1"](y_eval, y_pred, average="macro", zero_division=0)
    f1_weighted = sk["f1"](y_eval, y_pred, average="weighted", zero_division=0)
    prec_macro = sk["precision"](y_eval, y_pred, average="macro", zero_division=0)
    rec_macro = sk["recall"](y_eval, y_pred, average="macro", zero_division=0)

    if le is not None:
        labels = list(range(len(le.classes_)))
        target_names = [str(c) for c in le.classes_]
    else:
        labels = sorted(list(set(np.unique(y_eval)).union(set(np.unique(y_pred)))))
        target_names = [str(c) for c in labels]

    report = sk["report"](y_eval, y_pred, labels=labels, target_names=target_names, zero_division=0)

    return {
        "status": "SUCCESS",
        "type": "Classification",
        "n_samples": len(X_eval),
        "n_features": len(features),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "latency_ms": eval_latency_ms,
        "report": report
    }


def evaluate_regression_model(target_name: str, model_data: dict, df: pd.DataFrame, mode: str, sk: dict) -> dict:
    """Evaluate a single regression model."""
    model = model_data.get("model")
    features = model_data.get("features", [])

    if target_name not in df.columns:
        return {"status": "SKIPPED", "reason": f"Target column '{target_name}' not found in CSV."}

    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        return {"status": "SKIPPED", "reason": f"Missing {len(missing_feats)} feature columns (e.g. {missing_feats[:2]})."}

    valid_df = df.dropna(subset=[target_name]).copy()
    if valid_df.empty:
        return {"status": "SKIPPED", "reason": "No non-null rows for this target column."}

    X = valid_df[features]
    y_true = valid_df[target_name].values.astype(float)

    if mode == "split" and len(X) >= 10:
        _, X_eval, _, y_eval = sk["split"](X, y_true, test_size=0.2, random_state=42)
    else:
        X_eval, y_eval = X, y_true

    t0 = time.time()
    y_pred = model.predict(X_eval)
    eval_latency_ms = (time.time() - t0) * 1000

    mae = sk["mae"](y_eval, y_pred)
    rmse = float(np.sqrt(sk["mse"](y_eval, y_pred)))
    r2 = sk["r2"](y_eval, y_pred)
    mean_bias = float(np.mean(y_pred - y_eval))

    return {
        "status": "SUCCESS",
        "type": "Regression",
        "n_samples": len(X_eval),
        "n_features": len(features),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_bias": mean_bias,
        "latency_ms": eval_latency_ms
    }


def main():
    parser = argparse.ArgumentParser(description="Test GP Models One by One")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR),
                        help="Path to models directory (default: gp_models/)")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_FILE),
                        help="Path to evaluation dataset CSV")
    parser.add_argument("--target", type=str, default=None,
                        help="Filter by specific model or target name")
    parser.add_argument("--mode", choices=["split", "full"], default="split",
                        help="Evaluation mode: 'split' (20 percent held-out test split) or 'full' (100 percent dataset)")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Optional limit on rows to load for faster evaluation (e.g. --nrows 10000)")
    args = parser.parse_args()

    sk = load_dependencies()
    data_path = Path(args.data)
    models_dir = Path(args.models_dir)

    print("==========================================================================")
    print("                SEQUENTIAL GP MODEL EVALUATION TESTER                     ")
    print("==========================================================================")
    print(f" Models Directory : {models_dir}")
    print(f" Dataset Path     : {data_path}")
    print(f" Evaluation Mode  : {args.mode.upper()} {'(20% Test Split)' if args.mode == 'split' else '(Full Dataset)'}")
    if args.target:
        print(f" Target Filter    : {args.target}")
    print("--------------------------------------------------------------------------\n")

    if not data_path.exists():
        print(f"[ERROR] Dataset file not found: {data_path}")
        sys.exit(1)

    print("Loading test dataset...")
    df = pd.read_csv(data_path, nrows=args.nrows)
    print(f"Dataset successfully loaded: {len(df):,} rows, {len(df.columns)} columns.\n")

    model_files = find_model_files(models_dir, args.target)

    if not model_files:
        print(f"[WARNING] No .pkl model files found in '{models_dir}'.")
        print("Models are currently being trained. Re-run this script when models arrive in `gp_models/`.")
        sys.exit(0)

    print(f"Found {len(model_files)} model(s) to test sequentially.\n")

    clf_results = []
    reg_results = []
    skipped_count = 0

    for idx, pkl_path in enumerate(model_files, 1):
        filename = pkl_path.name
        print(f"[{idx}/{len(model_files)}] Testing model: {filename}")
        print("-" * (18 + len(filename)))

        try:
            model_data = sk["joblib"].load(pkl_path)
        except Exception as e:
            print(f"   ↳ [ERROR] Could not load pkl artifact: {e}")
            skipped_count += 1
            print()
            continue

        # Extract target name from filename
        stem = pkl_path.stem
        if stem.endswith("_rf_classifier") or stem.endswith("_gb_classifier"):
            target_name = stem.rsplit("_", 2)[0]
            is_clf = True
        elif stem.endswith("_rf_regressor") or stem.endswith("_gb_regressor"):
            target_name = stem.rsplit("_", 2)[0]
            is_clf = False
        else:
            target_name = stem
            model_obj = model_data.get("model") if isinstance(model_data, dict) else model_data
            is_clf = hasattr(model_obj, "predict_proba")

        if is_clf:
            res = evaluate_classification_model(target_name, model_data, df, args.mode, sk)
            res["filename"] = filename
            res["target"] = target_name

            if res["status"] == "SUCCESS":
                clf_results.append(res)
                print(f"   Type        : Classification")
                print(f"   Samples     : {res['n_samples']:,} | Features: {res['n_features']}")
                print(f"   Accuracy    : {res['accuracy']*100:.2f}%")
                print(f"   F1 (Macro)  : {res['f1_macro']:.4f}")
                print(f"   F1 (Weight) : {res['f1_weighted']:.4f}")
                print(f"   Precision   : {res['precision_macro']:.4f}")
                print(f"   Recall      : {res['recall_macro']:.4f}")
                print(f"   Inference   : {res['latency_ms']:.2f} ms")
            else:
                skipped_count += 1
                print(f"   ↳ [{res['status']}] {res['reason']}")
        else:
            res = evaluate_regression_model(target_name, model_data, df, args.mode, sk)
            res["filename"] = filename
            res["target"] = target_name

            if res["status"] == "SUCCESS":
                reg_results.append(res)
                print(f"   Type        : Regression")
                print(f"   Samples     : {res['n_samples']:,} | Features: {res['n_features']}")
                print(f"   R² Score    : {res['r2']:.4f}")
                print(f"   MAE         : {res['mae']:.4f}")
                print(f"   RMSE        : {res['rmse']:.4f}")
                print(f"   Mean Bias   : {res['mean_bias']:.4f}")
                print(f"   Inference   : {res['latency_ms']:.2f} ms")
            else:
                skipped_count += 1
                print(f"   ↳ [{res['status']}] {res['reason']}")

        print()  # Empty line separator for readability

    # Print Final Summary Tables
    print("=" * 80)
    print("                        SUMMARY OF ALL TESTED MODELS                         ")
    print("=" * 80)

    if clf_results:
        print("\n--- CLASSIFICATION MODELS ---")
        summary_clf = pd.DataFrame([
            {
                "Model File": r["filename"],
                "Target": r["target"],
                "Accuracy": f"{r['accuracy']*100:.2f}%",
                "F1 (Macro)": f"{r['f1_macro']:.4f}",
                "F1 (Weighted)": f"{r['f1_weighted']:.4f}",
                "Precision": f"{r['precision_macro']:.4f}",
                "Recall": f"{r['recall_macro']:.4f}",
            }
            for r in clf_results
        ])
        print(summary_clf.to_string(index=False))

    if reg_results:
        print("\n--- REGRESSION MODELS ---")
        summary_reg = pd.DataFrame([
            {
                "Model File": r["filename"],
                "Target": r["target"],
                "R² Score": f"{r['r2']:.4f}",
                "MAE": f"{r['mae']:.4f}",
                "RMSE": f"{r['rmse']:.4f}",
                "Mean Bias": f"{r['mean_bias']:.4f}",
            }
            for r in reg_results
        ])
        print(summary_reg.to_string(index=False))

    print(f"\nTested: {len(clf_results) + len(reg_results)} successful | Skipped/Failed: {skipped_count}")

    # Export Summary CSV & Full Text Report
    out_csv = models_dir / "gp_models_accuracy_summary.csv"
    all_rows = []
    for r in clf_results:
        all_rows.append({
            "filename": r["filename"],
            "target": r["target"],
            "type": "classification",
            "mode": args.mode,
            "n_samples": r["n_samples"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "f1_weighted": r["f1_weighted"],
            "precision_macro": r["precision_macro"],
            "recall_macro": r["recall_macro"],
            "mae": None, "rmse": None, "r2": None, "mean_bias": None
        })
    for r in reg_results:
        all_rows.append({
            "filename": r["filename"],
            "target": r["target"],
            "type": "regression",
            "mode": args.mode,
            "n_samples": r["n_samples"],
            "accuracy": None, "f1_macro": None, "f1_weighted": None,
            "precision_macro": None, "recall_macro": None,
            "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"], "mean_bias": r["mean_bias"]
        })

    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\n[SAVED] Summary CSV saved to: {out_csv}")

    out_report = models_dir / "gp_models_accuracy_report.txt"
    with out_report.open("w") as f:
        f.write("GP MODELS EVALUATION REPORT\n")
        f.write("===========================\n")
        f.write(f"Models Dir: {models_dir}\n")
        f.write(f"Dataset   : {data_path}\n")
        f.write(f"Eval Mode : {args.mode}\n\n")

        if clf_results:
            f.write("CLASSIFICATION RESULTS:\n")
            f.write("-----------------------\n")
            for r in clf_results:
                f.write(f"Model: {r['filename']} (Target: {r['target']})\n")
                f.write(f"Accuracy: {r['accuracy']*100:.2f}% | F1 Macro: {r['f1_macro']:.4f} | F1 Weighted: {r['f1_weighted']:.4f}\n")
                f.write("Classification Report:\n")
                f.write(r['report'])
                f.write("\n" + "-"*60 + "\n")

        if reg_results:
            f.write("REGRESSION RESULTS:\n")
            f.write("-------------------\n")
            for r in reg_results:
                f.write(f"Model: {r['filename']} (Target: {r['target']})\n")
                f.write(f"R²: {r['r2']:.4f} | MAE: {r['mae']:.4f} | RMSE: {r['rmse']:.4f} | Mean Bias: {r['mean_bias']:.4f}\n")
                f.write("-" * 60 + "\n")

    print(f"[SAVED] Detailed text report saved to: {out_report}\n")


if __name__ == "__main__":
    main()
