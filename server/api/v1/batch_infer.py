"""Batch inference endpoint — accepts pre-computed feature vectors, bypasses spatial lookup.

This is an ADDITIVE internal endpoint only. No existing endpoints are changed.
It is designed for the daily pipeline where feature vectors are already constructed
from GEE extractions + serving-parquet static features.

POST /api/v1/infer/batch
  Body: { "rows": [{feature dict}, ...], "targets": [...], "observation_month": "YYYY-MM" }
  Returns: per-row prediction results with the same TargetPrediction shape as /api/v1/predict

Security: API key is REQUIRED (same header as existing endpoint).
          This endpoint must NEVER be exposed to the browser directly.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from server.config import (
    API_VERSION,
    CONTRACT_VERSION,
    MODEL_TARGETS,
    MAX_TARGETS_PER_REQUEST,
)
from server.core.catalog import model_catalog
from server.core.model_loader import ModelUnavailable, model_manager
from server.errors import ServiceError

router = APIRouter(prefix="/api/v1/infer", tags=["batch-inference-v1"])

# Hard limit to prevent accidental memory exhaustion
MAX_BATCH_ROWS = 500

# The exact 75 feature names the models expect (verified from features_serving.parquet)
SERVING_FEATURE_NAMES: list[str] = [
    "elevation_m", "slope_degrees", "aspect_degrees", "distance_to_surface_water_m",
    "soil_cec_cmol_kg_0_30cm", "soil_clay_pct_0_30cm", "soil_sand_pct_0_30cm",
    "soil_silt_pct_0_30cm", "soil_soc_g_kg_0_30cm", "soil_ph_h2o_0_30cm",
    "surface_water_occurrence_pct", "surface_water_seasonality_months",
    "distance_to_road_km", "road_density_km_per_sqkm",
    "distance_to_railway_km", "railway_density_km_per_sqkm",
    "distance_to_river_km", "river_density_km_per_sqkm",
    "urban_fraction", "builtup_fraction", "cropland_fraction",
    "non_cropland_fraction", "permanent_water_fraction",
    "population_density", "valid_agriculture_mask",
    "chirps_precipitation_mm", "mean_temperature_c", "solar_radiation_mj_m2_day",
    "chirps_precipitation_mm_mean", "chirps_precipitation_mm_max",
    "chirps_precipitation_mm_min", "chirps_precipitation_mm_range",
    "chirps_precipitation_mm_cv",
    "era5_soil_moisture_m3_m3_mean", "era5_soil_moisture_m3_m3_max",
    "era5_soil_moisture_m3_m3_min", "era5_soil_moisture_m3_m3_cv",
    "mean_temperature_c_mean", "mean_temperature_c_max",
    "mean_temperature_c_min", "mean_temperature_c_range",
    "ndvi_median_mean", "ndvi_median_max", "ndvi_median_min",
    "ndvi_median_growing_season_mean",
    "ndwi_mcf_median_mean", "ndwi_mcf_median_max",
    "s1_vh_db_median_mean", "s1_vv_db_median_mean",
    "solar_radiation_mj_m2_day_mean", "solar_radiation_mj_m2_day_max",
    "data_month",
    "crop_area_pct_monsoon_rice", "crop_area_pct_dry_season_rice",
    "crop_area_pct_maize", "crop_area_pct_sugarcane", "crop_area_pct_cassava",
    "crop_area_pct_durian", "crop_area_pct_mangosteen", "crop_area_pct_longan",
    "crop_area_pct_mango", "crop_area_pct_chili", "crop_area_pct_tomato",
    "crop_area_pct_black_gram", "crop_area_pct_green_gram",
    "crop_area_pct_pigeon_pea", "crop_area_pct_groundnut",
    "crop_area_pct_sesame", "crop_area_pct_rubber",
    "region_ayeyawaddy", "region_bago", "region_magway",
    "region_mandalay", "region_sagaing", "region_yangon",
]


# ── Pydantic contracts ────────────────────────────────────────────────────────

YEAR_MONTH_RE = __import__("re").compile(r"^(20\d{2})-(0[1-9]|1[0-2])$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BatchInferRequest(_StrictModel):
    rows: list[dict[str, Any]] = Field(
        min_length=1, max_length=MAX_BATCH_ROWS,
        description="List of feature dicts (75 features each)",
    )
    targets: list[str] | None = Field(
        default=None, min_length=1, max_length=MAX_TARGETS_PER_REQUEST,
        description="Subset of model targets to run. Omit for all 40.",
    )
    include_all_targets: bool = Field(
        default=False,
        description="Run all 40 model targets for every row.",
    )
    observation_month: str | None = Field(
        default=None,
        description="YYYY-MM for data_month feature injection.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "BatchInferRequest":
        if self.include_all_targets and self.targets:
            raise ValueError("targets and include_all_targets cannot be combined")
        if not self.include_all_targets and not self.targets:
            raise ValueError("provide targets or set include_all_targets=true")
        if self.targets:
            unknown = sorted(set(self.targets) - set(MODEL_TARGETS))
            if unknown:
                raise ValueError(f"unknown targets: {unknown}")
        if self.observation_month and not YEAR_MONTH_RE.fullmatch(self.observation_month):
            raise ValueError("observation_month must use YYYY-MM format")
        return self


class RowPrediction(_StrictModel):
    value: float | int | str
    label: str | None
    unit: str
    task_type: str
    confidence: float | None
    confidence_kind: str | None
    probabilities: dict[str, float] | None
    model_version: str
    validation_status: str
    warnings: list[str]


class BatchRowResult(_StrictModel):
    row_index: int
    grid_id: str | None
    predictions: dict[str, RowPrediction]
    errors: dict[str, str]  # target → error message for failed targets


class BatchInferResponse(_StrictModel):
    api_version: str
    catalog_version: str
    total_rows: int
    successful_rows: int
    failed_rows: int
    results: list[BatchRowResult]
    execution_time_ms: float


# ── Internal helpers ──────────────────────────────────────────────────────────

def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _run_one_target(
    target: str,
    feature_row: dict[str, Any],
) -> RowPrediction:
    """Run a single model target against a pre-built feature dict.

    Mirrors the logic of _predict_target() in endpoints.py but returns
    a RowPrediction contract instead of a TargetPrediction (which requires
    provenance fields not available in batch mode).
    """
    artifact, metadata = model_manager.get_model(target)
    model = artifact["model"]
    feature_names = [str(f) for f in artifact["features"]]

    missing = [n for n in feature_names if n not in feature_row]
    if missing:
        raise ModelUnavailable(
            f"{target}: feature row is missing {len(missing)} required features: {missing[:5]}"
        )

    values: list[float] = []
    missing_values: list[str] = []
    invalid_values: list[str] = []
    for name in feature_names:
        try:
            v = float(feature_row[name])
        except (TypeError, ValueError):
            v = math.nan
        if math.isnan(v):
            missing_values.append(name)
        elif not math.isfinite(v):
            invalid_values.append(name)
        values.append(v)

    if invalid_values:
        raise ModelUnavailable(
            f"{target}: feature row contains infinite values: {invalid_values[:5]}"
        )

    if missing_values:
        try:
            allows_nan = bool(model.__sklearn_tags__().input_tags.allow_nan)
        except Exception:
            allows_nan = False
        if not allows_nan:
            raise ModelUnavailable(
                f"{target}: model does not support {len(missing_values)} missing values"
            )

    frame = pd.DataFrame([values], columns=feature_names)
    try:
        raw = _native(model.predict(frame)[0])
    except Exception as exc:
        raise ModelUnavailable(f"{target}: model execution failed") from exc

    task_type = str(metadata["task_type"])
    label: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None

    if task_type == "classification":
        le = artifact.get("label_encoder")
        decoded = raw
        if le is not None:
            try:
                decoded = _native(le.inverse_transform([int(raw)])[0])
            except Exception as exc:
                raise ModelUnavailable(f"{target}: label decoding failed") from exc
        label = str(_native(decoded))
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(frame)[0]
                classes = getattr(model, "classes_", range(len(proba)))
                decoded_classes: list[Any] = []
                for cls in classes:
                    cv = _native(cls)
                    if le is not None:
                        cv = _native(le.inverse_transform([int(cv)])[0])
                    decoded_classes.append(cv)
                probabilities = {
                    str(cv): float(p)
                    for cv, p in zip(decoded_classes, proba, strict=True)
                }
                confidence = max(probabilities.values()) if probabilities else None
            except Exception:
                pass

    return RowPrediction(
        value=_native(decoded) if task_type == "classification" else _native(raw),
        label=label,
        unit=str(metadata.get("unit", "")),
        task_type=task_type,
        confidence=confidence,
        confidence_kind="random_forest_vote_share_uncalibrated" if task_type == "classification" else None,
        probabilities=probabilities,
        model_version=str(metadata.get("model_version", "unknown")),
        validation_status=str(metadata.get("validation_status", "unknown")),
        warnings=list(metadata.get("warnings", [])),
    )


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/batch",
    response_model=BatchInferResponse,
    summary="Batch inference with pre-computed feature vectors",
    description=(
        "Accepts a list of 75-feature dicts and runs the requested model targets "
        "against each row WITHOUT spatial lookup. Designed for the daily pipeline "
        "where feature vectors are already constructed from GEE + serving-parquet data. "
        "Requires API key. Internal use only."
    ),
)
async def batch_infer(body: BatchInferRequest, request: Request) -> BatchInferResponse:
    t0 = time.perf_counter()

    # Resolve which targets to run
    targets: list[str]
    if body.include_all_targets:
        targets = list(MODEL_TARGETS)
    else:
        targets = list(body.targets or [])

    # Inject data_month from observation_month if provided and feature is missing
    data_month: int | None = None
    if body.observation_month:
        try:
            data_month = int(body.observation_month.split("-")[1])
        except (IndexError, ValueError):
            pass

    results: list[BatchRowResult] = []
    successful_rows = 0
    failed_rows = 0

    for idx, raw_row in enumerate(body.rows):
        feature_row = dict(raw_row)

        # Inject data_month if not already present
        if data_month is not None and "data_month" not in feature_row:
            feature_row["data_month"] = float(data_month)

        predictions: dict[str, RowPrediction] = {}
        errors: dict[str, str] = {}
        row_failed = False

        for target in targets:
            try:
                predictions[target] = _run_one_target(target, feature_row)
            except (ModelUnavailable, ServiceError, Exception) as exc:
                errors[target] = str(exc)
                row_failed = True

        grid_id = str(raw_row.get("grid_id", raw_row.get("index", ""))) or None

        results.append(BatchRowResult(
            row_index=idx,
            grid_id=grid_id,
            predictions=predictions,
            errors=errors,
        ))

        if row_failed and not predictions:
            failed_rows += 1
        else:
            successful_rows += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Fetch catalog version for the response header
    try:
        catalog = model_catalog.get_catalog()
        catalog_ver = catalog.get("catalog_version", "unknown")
    except Exception:
        catalog_ver = "unknown"

    return BatchInferResponse(
        api_version=API_VERSION,
        catalog_version=catalog_ver,
        total_rows=len(body.rows),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        results=results,
        execution_time_ms=round(elapsed_ms, 2),
    )
