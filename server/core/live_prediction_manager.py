#!/usr/bin/env python3
"""
server/core/live_prediction_manager.py — Live prediction database server cache manager
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from server.config import PROJECT_ROOT

LIVE_PREDICTIONS_PARQUET = PROJECT_ROOT / "data" / "processed" / "live_predictions.parquet"
LIVE_PREDICTIONS_CSV = PROJECT_ROOT / "data" / "processed" / "live_predictions.csv"

class LivePredictionManager:
    """
    Manages the live pre-computed predictions dataset:
    1. Loads predictions on demand / updates.
    2. Performs O(1) lookups by system:index.
    3. Performs O(log N) KD-Tree nearest neighbor lookup by (lat, lon).
    """
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.index_map: Dict[str, int] = {}
        self.kdtree: Optional[cKDTree] = None
        self.coordinates: Optional[np.ndarray] = None
        self.is_loaded: bool = False
        self.load_predictions()

    def load_predictions(self):
        """Loads or reloads the live predictions dataset."""
        target_path = None
        is_parquet = False

        if LIVE_PREDICTIONS_PARQUET.exists():
            target_path = LIVE_PREDICTIONS_PARQUET
            is_parquet = True
        elif LIVE_PREDICTIONS_CSV.exists():
            target_path = LIVE_PREDICTIONS_CSV
            is_parquet = False

        if not target_path or not target_path.exists():
            print("[LIVE PREDICTIONS] Prediction database not found. Falling back to real-time model inference.")
            self.is_loaded = False
            self.df = None
            return

        print(f"[LIVE PREDICTIONS] Loading pre-computed predictions database: {target_path.name}...")
        try:
            if is_parquet:
                self.df = pd.read_parquet(target_path)
            else:
                self.df = pd.read_csv(target_path)

            # Build index map
            if "system:index" in self.df.columns:
                self.index_map = {str(idx): i for i, idx in enumerate(self.df["system:index"].values)}
            else:
                # If there's no system:index column, use integer strings
                self.index_map = {str(i): i for i in range(len(self.df))}

            # Build coordinates KD-Tree if lat/lon are present
            lat_col = next((c for c in ["latitude", "lat", "y"] if c in self.df.columns), None)
            lon_col = next((c for c in ["longitude", "lon", "x"] if c in self.df.columns), None)

            if lat_col and lon_col:
                lats = self.df[lat_col].values.astype(float)
                lons = self.df[lon_col].values.astype(float)
                self.coordinates = np.column_stack((lats, lons))
                self.kdtree = cKDTree(self.coordinates)

            self.is_loaded = True
            print(f"[LIVE PREDICTIONS] Loaded {len(self.df):,} pre-computed predictions.")
        except Exception as e:
            print(f"[LIVE PREDICTIONS] Error loading predictions database: {e}")
            self.is_loaded = False

    def lookup_by_system_index(self, system_index: str) -> Optional[Dict[str, Any]]:
        """O(1) lookup by system:index."""
        if not self.is_loaded or self.df is None:
            return None
        row_idx = self.index_map.get(str(system_index))
        if row_idx is not None and row_idx < len(self.df):
            return self.df.iloc[row_idx].to_dict()
        return None

    def lookup_by_lat_lon(self, lat: float, lon: float) -> Optional[Tuple[Dict[str, Any], float]]:
        """O(log N) nearest neighbor coordinate lookup."""
        if not self.is_loaded or self.df is None or self.kdtree is None:
            return None
        try:
            dist, row_idx = self.kdtree.query([float(lat), float(lon)])
            row_dict = self.df.iloc[row_idx].to_dict()
            return row_dict, float(dist)
        except Exception:
            return None

# Singleton instance
live_prediction_manager = LivePredictionManager()
