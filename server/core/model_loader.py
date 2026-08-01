#!/usr/bin/env python3
"""
server/core/model_loader.py — Memory-Efficient LRU Model Loader & Prototype Fallback Manager
"""

import sys
import gc
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib

from server.config import MODELS_DIR, PROTOTYPES_DIR, MAX_LOADED_MODELS, BOOST_MODE

class LRUModelManager:
    """
    LRU (Least Recently Used) Model Manager with optional Boost Mode.
    - Normal Mode: Keeps model memory footprint within RAM limits (LRU eviction cap).
    - Boost Mode: Preloads & pins all models permanently in RAM for zero-latency (<1ms) inference.
    """

    def __init__(self, models_dir: Path = MODELS_DIR, prototypes_dir: Path = PROTOTYPES_DIR, max_models: int = MAX_LOADED_MODELS, boost_mode: bool = BOOST_MODE):
        self.models_dir = Path(models_dir)
        self.prototypes_dir = Path(prototypes_dir)
        self.max_models = max_models
        self.boost_mode = boost_mode
        
        # OrderedDict for LRU tracking: key = target_name, value = model_artifact_dict
        self._lru_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._prototype_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance & Health stats
        self.model_status_log: Dict[str, str] = {}
        self.load_times: Dict[str, float] = {}

        if self.boost_mode:
            print("[MODEL LOADER] BOOST_MODE environment setting detected! Preloading models...")
            self.preload_all_models()

    def _optimize_model(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizes the model for latency and memory by pruning trees and setting n_jobs=1."""
        if isinstance(artifact, dict) and "model" in artifact:
            model = artifact["model"]
            # Set n_jobs = 1 to optimize single-sample thread dispatching
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
            # Prune trees from 500 to 100 to reduce prediction latency (5x faster) and RAM footprint (4x smaller)
            if hasattr(model, "estimators_"):
                try:
                    if len(model.estimators_) > 100:
                        model.estimators_ = list(model.estimators_[:100])
                        if hasattr(model, "n_estimators"):
                            model.n_estimators = 100
                except Exception as e:
                    print(f"[MODEL LOADER WARN] Failed to prune estimators for model: {e}")
        return artifact

    def set_boost_mode(self, enabled: bool) -> Dict[str, Any]:
        """Dynamically enable or disable Boost Mode."""
        self.boost_mode = enabled
        if enabled:
            preloaded = self.preload_all_models()
            print(f"[MODEL LOADER 🚀] BOOST MODE ENABLED: {preloaded} models preloaded and pinned in RAM.")
            return {"status": "boost_mode_enabled", "models_preloaded": preloaded}
        else:
            print("[MODEL LOADER 🐢] BOOST MODE DISABLED: Reverting to standard LRU memory limits.")
            return {"status": "boost_mode_disabled", "lru_max_models_cap": self.max_models}

    def preload_all_models(self) -> int:
        """Preloads all available primary model artifacts into RAM."""
        preloaded_count = 0
        if not self.models_dir.exists():
            return 0

        for pkl in self.models_dir.glob("*.pkl"):
            target_name = pkl.stem
            for suffix in ["_rf_classifier", "_gb_classifier", "_rf_regressor", "_gb_regressor"]:
                if target_name.endswith(suffix):
                    target_name = target_name[:-len(suffix)]
                    break

            if target_name not in self._lru_cache:
                try:
                    artifact = joblib.load(pkl)
                    if isinstance(artifact, dict) and "model" in artifact:
                        self._lru_cache[target_name] = self._optimize_model(artifact)
                        preloaded_count += 1
                except Exception as e:
                    print(f"[MODEL LOADER WARN] Preload error for '{pkl.name}': {e}")
        return preloaded_count

    def _resolve_model_filename(self, target: str, directory: Path) -> Optional[Path]:
        """Finds matching .pkl file for a given target in directory."""
        if not directory.exists():
            return None
        
        # Direct target name check
        for pkl in directory.glob("*.pkl"):
            fname = pkl.stem
            clean_name = fname
            for suffix in ["_rf_classifier", "_gb_classifier", "_rf_regressor", "_gb_regressor"]:
                if fname.endswith(suffix):
                    clean_name = fname[:-len(suffix)]
                    break
            if clean_name == target or fname == target:
                return pkl
        return None

    def get_model(self, target: str, force_prototype: bool = False) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Retrieves model artifact for target.
        Returns (model_artifact_dict, source_type), where source_type is "primary" or "prototype".
        """
        # If prototype is explicitly forced (heavy load / memory pressure fallback)
        if force_prototype:
            proto_model = self._load_prototype_model(target)
            if proto_model:
                return proto_model, "prototype"

        # Check if already loaded in cache
        if target in self._lru_cache:
            self._lru_cache.move_to_end(target)   # Mark as recently used
            return self._lru_cache[target], "primary"

        # Try loading primary model from models/
        primary_path = self._resolve_model_filename(target, self.models_dir)
        if primary_path and primary_path.exists():
            # Check if cache is full (Skip eviction if Boost Mode is active)
            if not self.boost_mode and len(self._lru_cache) >= self.max_models:
                evicted_target, evicted_model = self._lru_cache.popitem(last=False)
                del evicted_model
                gc.collect()
                print(f"[MODEL LOADER] Memory Eviction: Unloaded '{evicted_target}' to keep RAM under limits.")

            start_t = time.time()
            try:
                artifact = joblib.load(primary_path)
                if isinstance(artifact, dict) and "model" in artifact:
                    self._lru_cache[target] = self._optimize_model(artifact)
                    self.load_times[target] = round(time.time() - start_t, 3)
                    print(f"[MODEL LOADER] Loaded primary model '{target}' in {self.load_times[target]}s")
                    return self._lru_cache[target], "primary"
            except Exception as e:
                print(f"[MODEL LOADER WARN] Failed to load primary model '{target}': {e}. Falling back to prototype.")

        # Fallback to prototype model
        proto_model = self._load_prototype_model(target)
        if proto_model:
            return proto_model, "prototype"

        return None, "none"

    def _load_prototype_model(self, target: str) -> Optional[Dict[str, Any]]:
        """Loads lightweight prototype model from models_prototypes/."""
        if target in self._prototype_cache:
            return self._prototype_cache[target]

        proto_path = self._resolve_model_filename(target, self.prototypes_dir)
        if proto_path and proto_path.exists():
            try:
                artifact = joblib.load(proto_path)
                if isinstance(artifact, dict) and "model" in artifact:
                    self._prototype_cache[target] = self._optimize_model(artifact)
                    print(f"[MODEL LOADER] Loaded fallback prototype model for '{target}' ({proto_path.name})")
                    return self._prototype_cache[target]
            except Exception as e:
                print(f"[MODEL LOADER ERROR] Failed to load prototype model for '{target}': {e}")
        return None

    def clear_cache(self):
        """Clears all cached models and releases RAM."""
        self._lru_cache.clear()
        self._prototype_cache.clear()
        gc.collect()


# Global singleton instance
model_manager = LRUModelManager()
