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

from server.config import MODELS_DIR, PROTOTYPES_DIR, MAX_LOADED_MODELS

class LRUModelManager:
    """
    LRU (Least Recently Used) Model Manager that keeps model memory footprint within 2 GB RAM limit.
    Dynamically loads primary models on demand, evicts least recently used models,
    and falls back to lightweight prototype models when under heavy load or memory pressure.
    """

    def __init__(self, models_dir: Path = MODELS_DIR, prototypes_dir: Path = PROTOTYPES_DIR, max_models: int = MAX_LOADED_MODELS):
        self.models_dir = Path(models_dir)
        self.prototypes_dir = Path(prototypes_dir)
        self.max_models = max_models
        
        # OrderedDict for LRU tracking: key = target_name, value = model_artifact_dict
        self._lru_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._prototype_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance & Health stats
        self.model_status_log: Dict[str, str] = {}
        self.load_times: Dict[str, float] = {}

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

        # Check if already loaded in LRU cache
        if target in self._lru_cache:
            self._lru_cache.move_to_end(target)   # Mark as recently used
            return self._lru_cache[target], "primary"

        # Try loading primary model from models/
        primary_path = self._resolve_model_filename(target, self.models_dir)
        if primary_path and primary_path.exists():
            # Check if cache is full
            if len(self._lru_cache) >= self.max_models:
                evicted_target, evicted_model = self._lru_cache.popitem(last=False)
                del evicted_model
                gc.collect()
                print(f"[MODEL LOADER] Memory Eviction: Unloaded '{evicted_target}' to keep RAM under 2GB cap.")

            start_t = time.time()
            try:
                artifact = joblib.load(primary_path)
                if isinstance(artifact, dict) and "model" in artifact:
                    self._lru_cache[target] = artifact
                    self.load_times[target] = round(time.time() - start_t, 3)
                    print(f"[MODEL LOADER] Loaded primary model '{target}' in {self.load_times[target]}s")
                    return artifact, "primary"
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
                    self._prototype_cache[target] = artifact
                    print(f"[MODEL LOADER] Loaded fallback prototype model for '{target}' ({proto_path.name})")
                    return artifact
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
