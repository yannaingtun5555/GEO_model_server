#!/usr/bin/env python3
"""Generate an immutable, checksum-backed catalog for every primary model artifact."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.config import MODEL_TARGETS
from server.model_metadata import TARGET_METADATA


MODEL_SUFFIXES = (
    "_rf_classifier.pkl",
    "_gb_classifier.pkl",
    "_rf_regressor.pkl",
    "_gb_regressor.pkl",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def record_for(df: pd.DataFrame, target: str) -> dict[str, Any]:
    if df.empty or "target" not in df:
        return {}
    rows = df[df["target"].astype(str) == target]
    if rows.empty:
        return {}
    return {str(key): json_value(value) for key, value in rows.iloc[0].to_dict().items()}


def find_artifact(models_dir: Path, target: str) -> Path:
    matches = [models_dir / f"{target}{suffix}" for suffix in MODEL_SUFFIXES]
    existing = [path for path in matches if path.is_file()]
    if len(existing) != 1:
        raise RuntimeError(
            f"{target}: expected exactly one primary artifact, found {[p.name for p in existing]}"
        )
    return existing[0]


def main() -> None:
    project_root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=project_root / "models")
    parser.add_argument(
        "--features",
        type=Path,
        default=project_root / "data" / "processed" / "features_serving.parquet",
    )
    parser.add_argument(
        "--spatial-index",
        type=Path,
        default=project_root / "data" / "processed" / "spatial_index.parquet",
    )
    parser.add_argument("--output", type=Path, default=project_root / "models" / "manifest.json")
    args = parser.parse_args()

    models_dir = args.models_dir.expanduser().resolve()
    features_path = args.features.expanduser().resolve()
    spatial_index_path = args.spatial_index.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (features_path, spatial_index_path):
        if not path.is_file():
            raise SystemExit(f"Required serving artifact is missing: {path}")

    features = pd.read_parquet(features_path)
    input_features = list(features.columns)
    input_schema_sha256 = hashlib.sha256(
        json.dumps(input_features, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    accuracy = pd.read_csv(models_dir / "test_accuracy_summary.csv")
    diagnostics_path = project_root / "model_testing" / "model_diagnostic_summary.csv"
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.is_file() else pd.DataFrame()

    models: list[dict[str, Any]] = []
    for target in MODEL_TARGETS:
        artifact_path = find_artifact(models_dir, target)
        artifact_sha256 = sha256_file(artifact_path)
        artifact = joblib.load(artifact_path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise RuntimeError(f"{artifact_path.name}: expected a model artifact dictionary")
        artifact_features = list(artifact.get("features", []))
        if artifact_features != input_features:
            raise RuntimeError(
                f"{target}: artifact feature order differs from the serving feature dataset"
            )

        metadata = TARGET_METADATA[target]
        task_type = metadata["task_type"]
        filename_task = "classification" if "classifier" in artifact_path.name else "regression"
        if task_type != filename_task:
            raise RuntimeError(
                f"{target}: metadata says {task_type}, artifact filename says {filename_task}"
            )

        label_encoder = artifact.get("label_encoder")
        classes = None
        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            classes = [json_value(value) for value in label_encoder.classes_]

        diagnostic = record_for(diagnostics, target)
        validation_status = str(diagnostic.get("overall_status", "unknown")).lower()
        if validation_status not in {"healthy", "flagged"}:
            validation_status = "unknown"
        warnings = [
            "Experimental surrogate model trained on rule-engineered labels; not field-validated."
        ]
        issue_summary = diagnostic.get("all_issues_summary")
        if validation_status == "flagged" and issue_summary:
            warnings.append(str(issue_summary))
        expected_classes = metadata.get("expected_classes")
        if classes is not None and expected_classes is not None:
            missing_classes = [value for value in expected_classes if value not in classes]
            if missing_classes:
                validation_status = "flagged"
                warnings.append(
                    f"Training artifact contains no examples for expected classes: {missing_classes}."
                )

        metrics = record_for(accuracy, target)
        metrics.pop("target", None)
        models.append(
            {
                "model_id": target,
                "display_name": metadata["display_name"],
                "group": metadata["group"],
                "task_type": task_type,
                "unit": metadata["unit"],
                "classes": classes,
                "expected_classes": expected_classes,
                "value_range": metadata.get("value_range"),
                "artifact_filename": artifact_path.name,
                "artifact_sha256": artifact_sha256,
                "artifact_size_bytes": artifact_path.stat().st_size,
                "model_version": f"sha256-{artifact_sha256[:12]}",
                "input_schema_sha256": input_schema_sha256,
                "input_feature_count": len(artifact_features),
                "model_source": "primary",
                "deployment_status": "experimental",
                "validation_status": validation_status,
                "field_validated": False,
                "label_source": "rule_engineered_surrogate",
                "probability_calibrated": False if task_type == "classification" else None,
                "metrics": metrics,
                "warnings": warnings,
            }
        )
        del artifact
        gc.collect()

    feature_dataset_sha256 = sha256_file(features_path)
    spatial_index_sha256 = sha256_file(spatial_index_path)
    catalog_digest_payload = {
        "schema_version": "model-catalog-v1",
        "feature_dataset_sha256": feature_dataset_sha256,
        "spatial_index_sha256": spatial_index_sha256,
        "models": [
            {
                key: model[key]
                for key in (
                    "model_id",
                    "task_type",
                    "unit",
                    "classes",
                    "expected_classes",
                    "value_range",
                    "artifact_sha256",
                    "input_schema_sha256",
                    "model_version",
                    "deployment_status",
                    "validation_status",
                    "field_validated",
                    "label_source",
                    "probability_calibrated",
                    "warnings",
                )
            }
            for model in models
        ],
    }
    catalog_version = hashlib.sha256(
        json.dumps(catalog_digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    manifest = {
        "schema_version": "model-catalog-v1",
        "catalog_version": catalog_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": "3.12",
            "scikit_learn": sklearn.__version__,
        },
        "feature_dataset": {
            "filename": features_path.name,
            "sha256": feature_dataset_sha256,
            "rows": len(features),
            "input_feature_count": len(input_features),
            "input_schema_sha256": input_schema_sha256,
        },
        "spatial_index": {
            "filename": spatial_index_path.name,
            "sha256": spatial_index_sha256,
            "rows": len(features),
            "schema_version": "spatial-index-v1",
        },
        "governance": {
            "label_source": "rule_engineered_surrogate",
            "field_validated": False,
            "validation_method": "random_train_test_split",
            "production_approval": False,
        },
        "models": models,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, output_path)
    print(
        f"Wrote catalog {catalog_version[:12]} with {len(models)} verified model artifacts "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
