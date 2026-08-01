#!/usr/bin/env python3
"""
run_all_model_tests.py — Comprehensive Diagnostic Suite Test Runner
===================================================================
Runs all model diagnostic tests across all 40 trained models in models/
against data/combined/combined_dataset.csv.

Exports:
  - CSV Summary : model_testing/model_diagnostic_summary.csv
  - Markdown    : model_testing/model_diagnostic_report.md
"""

import sys
import gc
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE    = PROJECT_ROOT / "data" / "combined" / "combined_dataset.csv"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUT_DIR   = PROJECT_ROOT / "model_testing"

# Import diagnostic test modules
sys.path.insert(0, str(PROJECT_ROOT))
from model_testing.test_overfitting import test_model_overfitting
from model_testing.test_data_leakage import test_model_data_leakage
from model_testing.test_class_imbalance import test_model_class_imbalance
from model_testing.test_edge_cases_bounds import test_model_edge_cases
from model_testing.test_residual_bias import test_model_residual_bias

def load_sklearn_dependencies():
    try:
        import joblib
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split, cross_val_score
        return {
            "joblib": joblib,
            "accuracy": accuracy_score,
            "f1": f1_score,
            "precision": precision_score,
            "recall": recall_score,
            "mse": mean_squared_error,
            "r2": r2_score,
            "split": train_test_split,
            "cv_score": cross_val_score
        }
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}", flush=True)
        sys.exit(1)

def main():
    print("=====================================================================", flush=True)
    print("      COMPREHENSIVE MACHINE LEARNING MODEL DIAGNOSTIC SUITE          ", flush=True)
    print("=====================================================================", flush=True)
    print(f" Dataset Path : {DATA_FILE}", flush=True)
    print(f" Models Dir   : {MODELS_DIR}", flush=True)
    print(f" Output Dir   : {OUTPUT_DIR}", flush=True)
    print("---------------------------------------------------------------------\n", flush=True)

    if not DATA_FILE.exists():
        print(f"[ERROR] Dataset file not found: {DATA_FILE}", flush=True)
        sys.exit(1)

    sk = load_sklearn_dependencies()

    print("Loading dataset...", flush=True)
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded dataset: {len(df):,} rows x {len(df.columns)} columns.\n", flush=True)

    model_files = sorted(list(MODELS_DIR.glob("*.pkl")))
    if not model_files:
        print(f"[ERROR] No .pkl model files found in {MODELS_DIR}", flush=True)
        sys.exit(1)

    print(f"Found {len(model_files)} trained model artifact(s) to evaluate.\n", flush=True)

    diagnostic_summary = []

    for idx, pkl_path in enumerate(model_files, 1):
        fname = pkl_path.stem
        if fname.endswith("_rf_classifier"):
            target = fname[:-14]
            is_clf = True
        elif fname.endswith("_rf_regressor"):
            target = fname[:-13]
            is_clf = False
        else:
            target = fname
            is_clf = True

        print(f"[{idx}/{len(model_files)}] Running diagnostics for '{target}' ({'Classification' if is_clf else 'Regression'})...", flush=True)

        try:
            model_data = sk["joblib"].load(pkl_path)
        except Exception as e:
            print(f"  [ERROR] Failed to load {pkl_path.name}: {e}", flush=True)
            continue

        # 1. Overfitting & Generalization Test
        res_overfit = test_model_overfitting(model_data, df, target, is_clf, sk)
        # 2. Data Leakage & Contamination Test
        res_leakage = test_model_data_leakage(model_data, df, target, is_clf, sk)
        # 3. Class Imbalance & Collapse Test
        res_imbalance = test_model_class_imbalance(model_data, df, target, is_clf, sk)
        # 4. Edge Cases & Boundary Violations Test
        res_edge = test_model_edge_cases(model_data, df, target, is_clf, sk)
        # 5. Residual Bias & Heteroscedasticity Test
        res_residual = test_model_residual_bias(model_data, df, target, is_clf, sk)

        # Collect flagged issues
        all_issues = []
        for r in [res_overfit, res_leakage, res_imbalance, res_edge, res_residual]:
            if r.get("status") == "FLAGGED":
                all_issues.append(r.get("issues"))

        overall_status = "HEALTHY" if len(all_issues) == 0 else "FLAGGED"

        # Log status
        if overall_status == "HEALTHY":
            print(f"  ✓ [{target}] PASSED all diagnostic checks (100% Healthy)", flush=True)
        else:
            print(f"  ⚠️  [{target}] FLAGGED ({len(all_issues)} issue(s)): {all_issues[0]}", flush=True)

        diagnostic_summary.append({
            "target": target,
            "type": "classification" if is_clf else "regression",
            "overall_status": overall_status,
            "issues_count": len(all_issues),
            "overfit_status": res_overfit.get("status"),
            "score_train": res_overfit.get("score_train"),
            "score_test": res_overfit.get("score_test"),
            "overfit_gap": res_overfit.get("gap"),
            "leakage_status": res_leakage.get("status"),
            "duplicate_overlap_pct": res_leakage.get("duplicate_overlap_pct"),
            "imbalance_status": res_imbalance.get("status"),
            "boundary_violations": res_edge.get("boundary_violations_count"),
            "residual_bias_status": res_residual.get("status"),
            "all_issues_summary": " | ".join(all_issues) if all_issues else "None (Optimal Model Health)"
        })

        del model_data
        gc.collect()

    summary_df = pd.DataFrame(diagnostic_summary)
    out_csv = OUTPUT_DIR / "model_diagnostic_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[SUMMARY] Saved diagnostic CSV summary to: {out_csv}", flush=True)

    # Write Markdown Diagnostic Report
    out_md = OUTPUT_DIR / "model_diagnostic_report.md"
    healthy_count = len(summary_df[summary_df["overall_status"] == "HEALTHY"])
    flagged_count = len(summary_df[summary_df["overall_status"] == "FLAGGED"])

    with out_md.open("w") as f:
        f.write("# Machine Learning Model Diagnostic & Health Report\n\n")
        f.write("## Executive Summary\n")
        f.write(f"- **Total Models Evaluated**: {len(summary_df)}\n")
        f.write(f"- **Healthy Models (Passed All Diagnostic Checks)**: **{healthy_count}** ({healthy_count/len(summary_df)*100:.1f}%)\n")
        f.write(f"- **Flagged Models (Requires Review/Refinement)**: **{flagged_count}** ({flagged_count/len(summary_df)*100:.1f}%)\n\n")

        f.write("## Overall Model Health Breakdown\n\n")
        f.write("| Target Model | Type | Overall Health | Issues Count | Train Score | Test Score | Generalization Gap | Issues Summary |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for r in diagnostic_summary:
            status_str = "🟢 HEALTHY" if r["overall_status"] == "HEALTHY" else "🟡 FLAGGED"
            train_str = f"{r['score_train']:.4f}" if r["score_train"] is not None else "N/A"
            test_str = f"{r['score_test']:.4f}" if r["score_test"] is not None else "N/A"
            gap_str = f"{r['overfit_gap']:.4f}" if r["overfit_gap"] is not None else "N/A"
            f.write(f"| `{r['target']}` | {r['type'].capitalize()} | {status_str} | {r['issues_count']} | {train_str} | {test_str} | {gap_str} | {r['all_issues_summary']} |\n")

        f.write("\n\n## Diagnostic Verification Methodology\n")
        f.write("1. **Overfitting Check**: Compares Train vs. Test score gap ($> 5\\%$ delta flags overfitting).\n")
        f.write("2. **Data Leakage Check**: Identifies target correlation ($|r| > 0.999$) and sample overlap.\n")
        f.write("3. **Class Imbalance Check**: Flags 0% precision/recall classes and baseline lift.\n")
        f.write("4. **Physical Boundaries Check**: Tests for negative yields or out-of-range physical values.\n")
        f.write("5. **Residual Bias Check**: Tests mean bias and error heteroscedasticity.\n")

    print(f"[REPORT] Saved detailed Markdown diagnostic report to: {out_md}\n", flush=True)

if __name__ == "__main__":
    main()
