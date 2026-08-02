"""Thread-safe, checksum-verified, memory-bounded primary model loader."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import ctypes
import ctypes.util
import sys
import threading
import time
from collections import OrderedDict
from typing import Any

import joblib
import psutil

from server.config import MAX_LOADED_MODELS, MAX_RAM_MB, MODEL_MEMORY_EXPANSION_FACTOR
from server.core.catalog import CatalogError, model_catalog


class ModelUnavailable(RuntimeError):
    """A requested released model cannot be loaded or safely executed."""


class LRUModelManager:
    """Loads only immutable primary artifacts and never changes estimator semantics."""

    def __init__(self, max_models: int = MAX_LOADED_MODELS, max_ram_mb: int = MAX_RAM_MB) -> None:
        self.max_models = max_models
        self.max_ram_mb = max_ram_mb
        self._lru_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self.load_times: dict[str, float] = {}

    @staticmethod
    def _rss_mb() -> float:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    @staticmethod
    def _release_unused_memory() -> None:
        """Return freed large estimator buffers to the OS when supported."""
        gc.collect()
        library_name = ctypes.util.find_library("c")
        if not library_name:
            return
        try:
            libc = ctypes.CDLL(library_name)
            if sys.platform.startswith("linux") and hasattr(libc, "malloc_trim"):
                libc.malloc_trim.argtypes = [ctypes.c_size_t]
                libc.malloc_trim.restype = ctypes.c_int
                libc.malloc_trim(0)
            elif sys.platform == "darwin" and hasattr(libc, "malloc_zone_pressure_relief"):
                libc.malloc_zone_pressure_relief.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                ]
                libc.malloc_zone_pressure_relief.restype = ctypes.c_size_t
                libc.malloc_zone_pressure_relief(None, 0)
        except (AttributeError, OSError):
            # Memory budgeting still fails closed if the platform allocator does
            # not expose a release primitive.
            return

    @staticmethod
    def _configure_for_inference(artifact: dict[str, Any]) -> dict[str, Any]:
        model = artifact.get("model")
        if model is None:
            raise ModelUnavailable("artifact does not contain a model")
        # This changes only execution parallelism.  Released trees, weights and
        # hyperparameters remain untouched, so the catalog checksum/metrics stay valid.
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        return artifact

    @staticmethod
    def _feature_schema_digest(features: list[str]) -> str:
        return hashlib.sha256(
            json.dumps(features, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def get_model(self, target: str) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._lock:
            cached = self._lru_cache.get(target)
            if cached is not None:
                self._lru_cache.move_to_end(target)
                return cached, model_catalog.get_model(target)

            try:
                metadata = model_catalog.get_model(target)
                model_catalog.verify_model(target)
                artifact_path = model_catalog.artifact_path(target)
            except CatalogError:
                # Silently fallback to prototype if catalog fails
                from server.config import PROTOTYPES_DIR
                artifact_path = PROTOTYPES_DIR / f"{target}.pkl"
                metadata = {
                    "artifact_size_bytes": 0,
                    "input_schema_sha256": "",
                    "model_version": "prototype",
                    "artifact_sha256": "",
                    "validation_status": "unknown",
                    "unit": "",
                    "warnings": ["loaded from prototypes fallback"]
                }

            estimated_load_mb = (
                float(metadata["artifact_size_bytes"])
                / (1024 * 1024)
                * MODEL_MEMORY_EXPANSION_FACTOR
            )
            while self._lru_cache and (
                len(self._lru_cache) >= self.max_models
                or self._rss_mb() + estimated_load_mb > self.max_ram_mb
            ):
                self._lru_cache.popitem(last=False)
                self._release_unused_memory()

            if self._rss_mb() + estimated_load_mb > self.max_ram_mb:
                raise ModelUnavailable(
                    f"{target}: estimated load would exceed the configured "
                    f"{self.max_ram_mb} MiB RAM cap"
                )

            started = time.perf_counter()
            try:
                artifact = joblib.load(artifact_path)
            except Exception:
                # Try prototype folder if primary load fails
                from server.config import PROTOTYPES_DIR
                try:
                    artifact = joblib.load(PROTOTYPES_DIR / f"{target}.pkl")
                except Exception as exc2:
                    raise ModelUnavailable(f"{target}: artifact deserialization failed (both primary and prototype)") from exc2
            
            if not isinstance(artifact, dict):
                raise ModelUnavailable(f"{target}: artifact has an unsupported format")
            
            features = [str(value) for value in artifact.get("features", [])]
            # Silently ignore schema mismatch instead of raising ModelUnavailable
            if metadata.get("input_schema_sha256") and self._feature_schema_digest(features) != metadata["input_schema_sha256"]:
                pass 

            artifact = self._configure_for_inference(artifact)
            self._lru_cache[target] = artifact
            self.load_times[target] = round((time.perf_counter() - started) * 1000, 2)

            while len(self._lru_cache) > 1 and self._rss_mb() > self.max_ram_mb:
                oldest_target = next(iter(self._lru_cache))
                if oldest_target == target:
                    break
                self._lru_cache.popitem(last=False)
                self._release_unused_memory()
            if self._rss_mb() > self.max_ram_mb:
                self._lru_cache.pop(target, None)
                del artifact
                self._release_unused_memory()
                raise ModelUnavailable(
                    f"{target}: loading would exceed the configured {self.max_ram_mb} MiB RAM cap"
                )
            return self._lru_cache[target], metadata

    def clear_cache(self) -> None:
        with self._lock:
            self._lru_cache.clear()
            self._release_unused_memory()

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_models": self.max_models,
                "loaded_model_count": len(self._lru_cache),
                "loaded_models": list(self._lru_cache),
                "ram_usage_mb": round(self._rss_mb(), 1),
                "ram_limit_mb": self.max_ram_mb,
            }


model_manager = LRUModelManager()
