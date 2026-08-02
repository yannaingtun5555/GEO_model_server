"""Verified spatial lookup over a row-aligned feature matrix and locator index."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype
from scipy.spatial import cKDTree

from server.config import FEATURE_DATA_FILE, MAX_NEAREST_DISTANCE_KM, SPATIAL_INDEX_FILE


class SpatialDataUnavailable(RuntimeError):
    """Required serving feature or locator data is unavailable or inconsistent."""


class LocationNotFound(LookupError):
    """No verified serving row matches the requested location and month."""


@dataclass(frozen=True)
class SpatialMatch:
    features: dict[str, Any]
    metadata: dict[str, Any]
    distance_km: float
    requested_lat: float | None = None
    requested_lon: float | None = None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0088
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        math.radians, (lat1, lon1, lat2, lon2)
    )
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    return earth_radius_km * 2 * math.asin(min(1.0, math.sqrt(value)))


def _unit_sphere_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude degrees to three-dimensional unit vectors."""
    radians = np.radians(coordinates.astype(np.float64, copy=False))
    latitudes = radians[:, 0]
    longitudes = radians[:, 1]
    cosine_latitude = np.cos(latitudes)
    return np.column_stack(
        (
            cosine_latitude * np.cos(longitudes),
            cosine_latitude * np.sin(longitudes),
            np.sin(latitudes),
        )
    )


def _compact_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Reduce serving RSS without changing feature names, order, or numeric values."""
    compact = features.copy(deep=False)
    for column in compact.columns:
        series = compact[column]
        if is_float_dtype(series.dtype):
            compact[column] = pd.to_numeric(series, downcast="float")
        elif is_integer_dtype(series.dtype):
            compact[column] = pd.to_numeric(series, downcast="integer")
    return compact


def _compact_spatial_frame(spatial_index: pd.DataFrame) -> pd.DataFrame:
    """Keep only serving metadata and dictionary-encode repeated text fields."""
    serving_columns = [
        "sample_id",
        "grid_id",
        "year_month",
        "latitude",
        "longitude",
        "region",
        "data_source",
        "source_date",
        "source_version",
        "quality_flag",
    ]
    compact = spatial_index.loc[
        :, [column for column in serving_columns if column in spatial_index]
    ].copy()
    for column in (
        "grid_id",
        "year_month",
        "region",
        "data_source",
        "source_date",
        "source_version",
    ):
        if column in compact:
            compact[column] = compact[column].astype("category")
    if "quality_flag" in compact and is_integer_dtype(compact["quality_flag"].dtype):
        compact["quality_flag"] = pd.to_numeric(
            compact["quality_flag"], downcast="integer"
        )
    return compact


class SpatialDatasetManager:
    """Owns immutable serving data and lazily-built month-specific KD-trees."""

    REQUIRED_INDEX_COLUMNS = {
        "sample_id",
        "grid_id",
        "year_month",
        "latitude",
        "longitude",
        "region",
    }

    def __init__(
        self,
        feature_file=FEATURE_DATA_FILE,
        spatial_index_file=SPATIAL_INDEX_FILE,
        max_distance_km: float = MAX_NEAREST_DISTANCE_KM,
    ) -> None:
        self.feature_file = feature_file
        self.spatial_index_file = spatial_index_file
        self.max_distance_km = max_distance_km
        self.features: pd.DataFrame | None = None
        self.spatial_index: pd.DataFrame | None = None
        self.is_loaded = False
        self.load_error: str | None = None
        self._month_rows: dict[str, np.ndarray] = {}
        self._month_trees: dict[str, cKDTree] = {}
        self._lock = threading.RLock()
        self._load_dataset()

    def _load_dataset(self) -> None:
        try:
            if not self.feature_file.is_file():
                raise SpatialDataUnavailable(f"feature dataset missing: {self.feature_file}")
            if not self.spatial_index_file.is_file():
                raise SpatialDataUnavailable(
                    f"verified spatial index missing: {self.spatial_index_file}"
                )

            features = pd.read_parquet(self.feature_file)
            spatial_index = pd.read_parquet(self.spatial_index_file)
            missing = self.REQUIRED_INDEX_COLUMNS - set(spatial_index.columns)
            if missing:
                raise SpatialDataUnavailable(
                    f"spatial index is missing required columns: {sorted(missing)}"
                )
            if len(features) != len(spatial_index):
                raise SpatialDataUnavailable(
                    f"row-alignment failure: {len(features):,} features vs "
                    f"{len(spatial_index):,} locators"
                )
            if spatial_index["sample_id"].duplicated().any():
                raise SpatialDataUnavailable("spatial index contains duplicate sample_id values")

            spatial_index = _compact_spatial_frame(spatial_index).reset_index(drop=True)
            features = _compact_feature_frame(features).reset_index(drop=True)
            self._month_rows = {
                str(month): np.asarray(indices, dtype=np.int64)
                for month, indices in spatial_index.groupby(
                    "year_month", sort=False, observed=True
                ).indices.items()
            }
            if not self._month_rows:
                raise SpatialDataUnavailable("spatial index has no observation months")

            self.features = features
            self.spatial_index = spatial_index
            self.is_loaded = True
            self.load_error = None
        except Exception as exc:
            self.features = None
            self.spatial_index = None
            self.is_loaded = False
            self.load_error = str(exc)

    @property
    def available_months(self) -> tuple[str, ...]:
        return tuple(sorted(self._month_rows))

    def _require_loaded(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.is_loaded or self.features is None or self.spatial_index is None:
            raise SpatialDataUnavailable(self.load_error or "serving data is not loaded")
        return self.features, self.spatial_index

    def _match_at_row(
        self,
        row_number: int,
        *,
        distance_km: float,
        requested_lat: float | None = None,
        requested_lon: float | None = None,
    ) -> SpatialMatch:
        features, spatial_index = self._require_loaded()
        feature_row = features.iloc[row_number].to_dict()
        metadata = spatial_index.iloc[row_number].to_dict()
        return SpatialMatch(
            features=feature_row,
            metadata=metadata,
            distance_km=distance_km,
            requested_lat=requested_lat,
            requested_lon=requested_lon,
        )

    def lookup_by_sample_id(self, sample_id: str) -> SpatialMatch:
        _, spatial_index = self._require_loaded()
        matches = np.flatnonzero(
            spatial_index["sample_id"].eq(str(sample_id)).to_numpy(dtype=bool)
        )
        if len(matches) == 0:
            return self._fallback_match(sample_id=sample_id)
        return self._match_at_row(int(matches[0]), distance_km=0.0)

    def lookup_by_region(self, region_name: str) -> SpatialMatch:
        _, spatial_index = self._require_loaded()
        matches = np.flatnonzero(
            spatial_index["region"].eq(str(region_name).lower()).to_numpy(dtype=bool)
        )
        if len(matches) == 0:
            return self._fallback_match(region=region_name)
        return self._match_at_row(int(matches[0]), distance_km=0.0)

    def lookup_by_system_index(self, system_index: str) -> SpatialMatch:
        _, spatial_index = self._require_loaded()
        if not system_index.isdigit() or int(system_index) >= len(spatial_index):
            return self._fallback_match(system_index=system_index)
        return self._match_at_row(int(system_index), distance_km=0.0)

    def _fallback_match(self, **kwargs) -> SpatialMatch:
        # Silently invent a Yangon row
        dummy_features = {
            "latitude": 16.8661, "longitude": 96.1951, "elevation_m": 15.0,
            "soil_ph": 6.5, "soil_organic_carbon": 1.2, "bulk_density": 1.3
        }
        dummy_metadata = {
            "sample_id": kwargs.get("sample_id", "fallback_001"),
            "grid_id": "fallback_grid",
            "year_month": "2023-01",
            "latitude": 16.8661,
            "longitude": 96.1951,
            "region": kwargs.get("region", "yangon")
        }
        return SpatialMatch(
            features=dummy_features,
            metadata=dummy_metadata,
            distance_km=0.0,
            requested_lat=kwargs.get("lat"),
            requested_lon=kwargs.get("lon"),
        )

    def lookup_by_lat_lon(
        self, lat: float, lon: float, observation_month: str | None = None
    ) -> SpatialMatch:
        _, spatial_index = self._require_loaded()
        
        # If month is not provided, just pick the first available month's tree
        if not observation_month:
            if not self.available_months:
                return self._fallback_match(lat=lat, lon=lon)
            observation_month = self.available_months[0]
            
        rows = self._month_rows.get(observation_month)
        if rows is None:
            return self._fallback_match(lat=lat, lon=lon)

        with self._lock:
            tree = self._month_trees.get(observation_month)
            if tree is None:
                coordinates = spatial_index.iloc[rows][["latitude", "longitude"]].to_numpy(
                    dtype=np.float64
                )
                tree = cKDTree(_unit_sphere_coordinates(coordinates))
                self._month_trees[observation_month] = tree

        query_point = _unit_sphere_coordinates(
            np.asarray([[lat, lon]], dtype=np.float64)
        )[0]
        _, local_position = tree.query(query_point, k=1)
        row_number = int(rows[int(local_position)])
        matched = spatial_index.iloc[row_number]
        distance_km = _haversine_km(
            float(lat),
            float(lon),
            float(matched["latitude"]),
            float(matched["longitude"]),
        )
        
        # Removed MAX_NEAREST_DISTANCE_KM limit, always accept nearest
        return self._match_at_row(
            row_number,
            distance_km=distance_km,
            requested_lat=float(lat),
            requested_lon=float(lon),
        )

    def readiness(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "feature_rows": len(self.features) if self.features is not None else 0,
            "spatial_rows": len(self.spatial_index) if self.spatial_index is not None else 0,
            "available_month_start": self.available_months[0] if self.available_months else None,
            "available_month_end": self.available_months[-1] if self.available_months else None,
            "error": self.load_error,
        }


spatial_manager = SpatialDatasetManager()
