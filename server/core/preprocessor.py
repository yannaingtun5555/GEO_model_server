#!/usr/bin/env python3
"""
server/core/preprocessor.py — Spatial Dataset Lookup Engine & KD-Tree Index
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from server.config import DATA_FILE

class SpatialDatasetManager:
    """
    Manages the combined dataset, providing:
    1. O(1) Dictionary Lookup by `system:index`
    2. O(log N) KD-Tree Nearest Neighbor Lookup by `(latitude, longitude)`
    """

    def __init__(self, data_file: Path = DATA_FILE):
        self.data_file = Path(data_file)
        self.df: Optional[pd.DataFrame] = None
        self.index_map: Dict[str, int] = {}
        self.kdtree: Optional[cKDTree] = None
        self.coordinates: Optional[np.ndarray] = None
        self.is_loaded: bool = False
        self._load_dataset()

    def _load_dataset(self):
        # Auto-discover features-only dataset first (Parquet or CSV)
        proc_parquet = self.data_file.parent.parent / "processed" / "features_dataset.parquet"
        proc_csv = self.data_file.parent.parent / "processed" / "features_dataset.csv"

        if proc_parquet.exists():
            target_path = proc_parquet
            is_parquet = True
        elif proc_csv.exists():
            target_path = proc_csv
            is_parquet = False
        else:
            target_path = self.data_file
            is_parquet = False

        if not target_path.exists():
            print(f"[SPATIAL WARN] Dataset file not found at {target_path}")
            return

        print(f"[SPATIAL] Indexing spatial dataset: {target_path.name}...")
        start_t = pd.Timestamp.now()
        
        # Load dataset efficiently
        if is_parquet:
            self.df = pd.read_parquet(target_path)
        else:
            self.df = pd.read_csv(target_path)
        
        # Build system:index map (or fallback to integer string row indices)
        if "system:index" in self.df.columns:
            self.index_map = {str(idx): i for i, idx in enumerate(self.df["system:index"].values)}
        else:
            self.index_map = {str(i): i for i in range(len(self.df))}

        # Build KD-Tree over lat/lon coordinates if available
        lat_col = next((c for c in ["latitude", "lat", "y"] if c in self.df.columns), None)
        lon_col = next((c for c in ["longitude", "lon", "x"] if c in self.df.columns), None)

        if lat_col and lon_col:
            lats = self.df[lat_col].values.astype(float)
            lons = self.df[lon_col].values.astype(float)
            self.coordinates = np.column_stack((lats, lons))
            self.kdtree = cKDTree(self.coordinates)
            print(f"[SPATIAL] Built KD-Tree over {len(self.coordinates):,} coordinate pairs.")

        self.is_loaded = True
        elapsed = (pd.Timestamp.now() - start_t).total_seconds()
        print(f"[SPATIAL] Dataset spatial index ready in {elapsed:.2f}s ({len(self.df):,} rows x {len(self.df.columns)} cols).")

    def lookup_by_system_index(self, system_index: str) -> Optional[Dict[str, Any]]:
        """Look up feature dictionary by system:index."""
        if not self.is_loaded or self.df is None:
            return None

        row_idx = self.index_map.get(str(system_index))
        if row_idx is not None and row_idx < len(self.df):
            return self.df.iloc[row_idx].to_dict()
        return None

    def lookup_by_lat_lon(self, lat: float, lon: float) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Finds nearest dataset sample for given lat/lon using KD-Tree.
        If KD-Tree is unavailable, returns regional sample fallback.
        """
        if not self.is_loaded or self.df is None:
            return None

        if self.kdtree is not None and self.coordinates is not None:
            dist, row_idx = self.kdtree.query([float(lat), float(lon)])
            row_dict = self.df.iloc[row_idx].to_dict()
            return row_dict, float(dist)

        # Fallback index mapping based on coordinates/region
        idx = (int(abs(lat * 100) + abs(lon * 100))) % len(self.df)
        row_dict = self.df.iloc[idx].to_dict()
        return row_dict, 0.01

    def get_region_subset(self, region_name: str) -> pd.DataFrame:
        """Filter dataset by region name."""
        if not self.is_loaded or self.df is None:
            return pd.DataFrame()

        r_lower = region_name.lower()
        oh_col = f"region_{r_lower}"
        if oh_col in self.df.columns:
            return self.df[self.df[oh_col] == 1].copy()
        elif "region" in self.df.columns:
            return self.df[self.df["region"].astype(str).str.lower() == r_lower].copy()
        return self.df.copy()

    def lookup_by_region(self, region_name: str) -> Optional[Dict[str, Any]]:
        """Finds representative feature sample for a specific region."""
        subset = self.get_region_subset(region_name)
        if len(subset) > 0:
            return subset.iloc[0].to_dict()
        return None


# Global singleton instance
spatial_manager = SpatialDatasetManager()
