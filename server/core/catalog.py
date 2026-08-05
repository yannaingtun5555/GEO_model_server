"""Immutable model catalog and artifact integrity checks."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from server.config import (
    FEATURE_DATA_FILE,
    MODEL_MANIFEST_FILE,
    MODELS_DIR,
    MODEL_TARGETS,
    SPATIAL_INDEX_FILE,
    VERIFY_MODEL_CHECKSUMS_ON_STARTUP,
)


class CatalogError(RuntimeError):
    """The model catalog or one of its declared artifacts is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelCatalog:
    def __init__(self, manifest_file: Path = MODEL_MANIFEST_FILE) -> None:
        self.manifest_file = Path(manifest_file)
        self.manifest: dict[str, Any] = {}
        self.models: dict[str, dict[str, Any]] = {}
        self.load_error: str | None = None
        self._verified_models: set[str] = set()
        self._serving_data_verified = False
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        try:
            if not self.manifest_file.is_file():
                raise CatalogError(f"model manifest missing: {self.manifest_file}")
            manifest = json.loads(self.manifest_file.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != "model-catalog-v1":
                raise CatalogError("unsupported model catalog schema")
            models = manifest.get("models")
            if not isinstance(models, list):
                raise CatalogError("model catalog models must be an array")
            by_id = {str(model.get("model_id")): model for model in models}
            if set(by_id) != set(MODEL_TARGETS):
                raise CatalogError(
                    "catalog targets differ from the canonical 40-target contract"
                )
            for target, model in by_id.items():
                digest = str(model.get("artifact_sha256", ""))
                if len(digest) != 64:
                    raise CatalogError(f"{target}: invalid artifact SHA-256")
                if model.get("deployment_status") != "experimental":
                    raise CatalogError(f"{target}: unsupported deployment status")
                if model.get("task_type") == "classification":
                    if model.get("probability_calibrated") is not False:
                        raise CatalogError(
                            f"{target}: classifier calibration status must be explicit"
                        )
                elif model.get("probability_calibrated") is not None:
                    raise CatalogError(
                        f"{target}: regression cannot declare probability calibration"
                    )
            self.manifest = manifest
            self.models = by_id
            self.load_error = None
            if VERIFY_MODEL_CHECKSUMS_ON_STARTUP:
                self.verify_serving_data()
                for target in MODEL_TARGETS:
                    self.verify_model(target)
        except Exception as exc:
            self.manifest = {}
            self.models = {}
            self.load_error = str(exc)

    @property
    def catalog_version(self) -> str:
        return str(self.manifest.get("catalog_version", "unavailable"))

    @property
    def feature_dataset_sha256(self) -> str:
        return str(self.manifest.get("feature_dataset", {}).get("sha256", ""))

    @property
    def spatial_index_sha256(self) -> str:
        return str(self.manifest.get("spatial_index", {}).get("sha256", ""))

    @property
    def production_approved(self) -> bool:
        return self.manifest.get("governance", {}).get("production_approval") is True

    def get_model(self, target: str) -> dict[str, Any]:
        model = self.models.get(target)
        if model is None:
            raise CatalogError(f"unknown model target: {target}")
        return model

    def artifact_path(self, target: str) -> Path:
        model = self.get_model(target)
        filename = str(model["artifact_filename"])
        path = (MODELS_DIR / filename).resolve()
        models_root = MODELS_DIR.resolve()
        if models_root not in path.parents:
            raise CatalogError(f"{target}: artifact path escaped MODELS_DIR")
        return path

    def verify_model(self, target: str) -> None:
        with self._lock:
            if target in self._verified_models:
                return
            path = self.artifact_path(target)
            if not path.is_file():
                raise CatalogError(f"{target}: primary artifact is missing")
            if VERIFY_MODEL_CHECKSUMS_ON_STARTUP:
                expected = str(self.models[target]["artifact_sha256"])
                actual = sha256_file(path)
                if actual != expected:
                    raise CatalogError(f"{target}: artifact checksum mismatch")
            self._verified_models.add(target)

    def verify_serving_data(self) -> None:
        declared = (
            (FEATURE_DATA_FILE, self.feature_dataset_sha256, "feature dataset"),
            (SPATIAL_INDEX_FILE, self.spatial_index_sha256, "spatial index"),
        )
        for path, expected, label in declared:
            if not path.is_file():
                raise CatalogError(f"{label} is missing: {path}")
            if VERIFY_MODEL_CHECKSUMS_ON_STARTUP:
                if not expected or sha256_file(path) != expected:
                    raise CatalogError(f"{label} checksum mismatch")
        self._serving_data_verified = True

    def verify_release(self) -> None:
        """Verify every serving checksum once before declaring readiness."""
        with self._lock:
            if not self._serving_data_verified:
                self.verify_serving_data()
            for target in MODEL_TARGETS:
                self.verify_model(target)

    def list_models(self) -> list[dict[str, Any]]:
        result = []
        for target in MODEL_TARGETS:
            item = dict(self.models[target])
            item.pop("artifact_filename", None)
            item.pop("expected_classes", None)
            item.pop("group", None)
            item.pop("input_feature_count", None)
            path = self.artifact_path(target)
            item["ready"] = path.is_file() and target in self._verified_models
            result.append(item)
        return result

    def readiness(self) -> dict[str, Any]:
        available = sum(1 for target in self.models if self.artifact_path(target).is_file())
        return {
            "loaded": bool(self.models),
            "catalog_version": self.catalog_version,
            "declared_model_count": len(self.models),
            "available_model_count": available,
            "required_model_count": len(MODEL_TARGETS),
            "verified_model_count": len(self._verified_models),
            "serving_data_verified": self._serving_data_verified,
            "error": self.load_error,
        }


model_catalog = ModelCatalog()
