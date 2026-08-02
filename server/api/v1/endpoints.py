"""Strict, fail-closed model-serving API v1 endpoints."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request

from server.config import (
    API_VERSION,
    CONTRACT_VERSION,
    MAX_EXPANDED_SYNC_TARGETS,
    MODEL_TARGETS,
)
from server.contracts import PredictionRequest, PredictionResponse
from server.core.cache import cache_manager
from server.core.catalog import CatalogError, model_catalog
from server.core.model_loader import ModelUnavailable, model_manager
from server.core.preprocessor import (
    LocationNotFound,
    SpatialDataUnavailable,
    SpatialMatch,
    spatial_manager,
)
from server.core.request_queue import ExecutionTimeout, QueueTimeout, request_queue
from server.errors import ServiceError
from server.services.composite_features import (
    COMPOSITE_DEPENDENCIES,
    CompositeFeaturesEngine,
    resolve_targets,
)
from server.model_metadata import TARGET_METADATA


router = APIRouter(prefix="/api/v1", tags=["model-serving-v1"])


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _optional_string(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _predict_target(target: str, feature_row: dict[str, Any]) -> dict[str, Any]:
    artifact, metadata = model_manager.get_model(target)
    model = artifact["model"]
    feature_names = [str(value) for value in artifact["features"]]
    missing = [name for name in feature_names if name not in feature_row]
    if missing:
        raise ModelUnavailable(f"{target}: serving row is missing {len(missing)} model features")

    values: list[float] = []
    missing_values: list[str] = []
    invalid_values: list[str] = []
    for name in feature_names:
        try:
            value = float(feature_row[name])
        except (TypeError, ValueError):
            value = math.nan
        if math.isnan(value):
            missing_values.append(name)
        elif not math.isfinite(value):
            invalid_values.append(name)
        values.append(value)
    if invalid_values:
        raise ModelUnavailable(
            f"{target}: serving row contains infinite values for {len(invalid_values)} features"
        )
    if missing_values:
        try:
            allows_nan = bool(model.__sklearn_tags__().input_tags.allow_nan)
        except Exception:
            allows_nan = False
        if not allows_nan:
            raise ModelUnavailable(
                f"{target}: model does not support {len(missing_values)} missing input values"
            )

    input_frame = pd.DataFrame([values], columns=feature_names)
    try:
        raw_prediction = _native(model.predict(input_frame)[0])
    except Exception as exc:
        raise ModelUnavailable(f"{target}: model execution failed") from exc

    task_type = str(metadata["task_type"])
    label: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    if task_type == "classification":
        label_encoder = artifact.get("label_encoder")
        decoded = raw_prediction
        if label_encoder is not None:
            try:
                decoded = _native(label_encoder.inverse_transform([int(raw_prediction)])[0])
            except Exception as exc:
                raise ModelUnavailable(f"{target}: classifier label decoding failed") from exc
        value: float | int | str = _native(decoded)
        label = str(value)
        if hasattr(model, "predict_proba"):
            try:
                probability_values = model.predict_proba(input_frame)[0]
                model_classes = getattr(model, "classes_", range(len(probability_values)))
                decoded_classes: list[Any] = []
                for model_class in model_classes:
                    class_value: Any = _native(model_class)
                    if label_encoder is not None:
                        class_value = _native(
                            label_encoder.inverse_transform([int(class_value)])[0]
                        )
                    decoded_classes.append(class_value)
                probabilities = {
                    str(class_value): float(probability)
                    for class_value, probability in zip(
                        decoded_classes, probability_values, strict=True
                    )
                }
                confidence = max(probabilities.values()) if probabilities else None
            except Exception as exc:
                raise ModelUnavailable(f"{target}: classifier probability decoding failed") from exc

        declared_classes = metadata.get("classes")
        semantic_classes = metadata.get("expected_classes")
        if declared_classes is not None and value not in declared_classes:
            raise ModelUnavailable(f"{target}: model returned an undeclared class")
        if semantic_classes is not None and value not in semantic_classes:
            raise ModelUnavailable(f"{target}: model returned an unsupported semantic class")
        if probabilities is not None:
            expected_probability_keys = {
                str(class_value) for class_value in (declared_classes or [])
            }
            if set(probabilities) != expected_probability_keys:
                raise ModelUnavailable(f"{target}: probability classes differ from the catalog")
            if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ModelUnavailable(f"{target}: classifier probabilities do not sum to one")
        confidence_kind = "random_forest_vote_share_uncalibrated"
    else:
        value = float(raw_prediction)
        if not math.isfinite(value):
            raise ModelUnavailable(f"{target}: model returned a non-finite value")
        value_range = TARGET_METADATA[target].get("value_range")
        if value_range is not None:
            minimum, maximum = value_range
            if minimum is not None and value < float(minimum):
                raise ModelUnavailable(f"{target}: prediction is below its declared range")
            if maximum is not None and value > float(maximum):
                raise ModelUnavailable(f"{target}: prediction is above its declared range")
        confidence_kind = None

    warnings = list(metadata["warnings"])
    if missing_values:
        warnings.append(
            f"The serving row contains {len(missing_values)} missing feature value(s); "
            "the released estimator's native missing-value handling was used."
        )

    return {
        "value": value,
        "label": label,
        "unit": metadata["unit"],
        "task_type": task_type,
        "confidence": confidence,
        "confidence_kind": confidence_kind,
        "probabilities": probabilities,
        "model_version": metadata["model_version"],
        "artifact_sha256": metadata["artifact_sha256"],
        "input_schema_sha256": metadata["input_schema_sha256"],
        "model_source": "primary",
        "deployment_status": "experimental",
        "validation_status": metadata["validation_status"],
        "warnings": warnings,
    }


def _location_payload(match: SpatialMatch) -> dict[str, Any]:
    metadata = match.metadata
    return {
        "sample_id": str(metadata["sample_id"]),
        "grid_id": str(metadata["grid_id"]),
        "region": str(metadata["region"]),
        "observation_month": str(metadata["year_month"]),
        "requested_lat": match.requested_lat,
        "requested_lon": match.requested_lon,
        "matched_lat": float(metadata["latitude"]),
        "matched_lon": float(metadata["longitude"]),
        "distance_km": round(match.distance_km, 4),
    }


def _provenance_payload(match: SpatialMatch) -> dict[str, Any]:
    metadata = match.metadata
    quality = metadata.get("quality_flag")
    if quality is None or (isinstance(quality, float) and math.isnan(quality)):
        quality_flag = None
    else:
        quality_flag = int(quality)
    return {
        "feature_dataset_sha256": model_catalog.feature_dataset_sha256,
        "spatial_index_sha256": model_catalog.spatial_index_sha256,
        "data_source": _optional_string(metadata.get("data_source")),
        "source_date": _optional_string(metadata.get("source_date")),
        "source_version": _optional_string(metadata.get("source_version")),
        "quality_flag": quality_flag,
        "label_source": "rule_engineered_surrogate",
        "field_validated": False,
    }


def _compute_prediction(req: PredictionRequest, request_id: str) -> dict[str, Any]:
    try:
        if req.sample_id is not None:
            match = spatial_manager.lookup_by_sample_id(req.sample_id)
        elif req.region_name is not None:
            match = spatial_manager.lookup_by_region(req.region_name)
        elif req.system_index is not None:
            match = spatial_manager.lookup_by_system_index(req.system_index)
        else:
            match = spatial_manager.lookup_by_lat_lon(
                float(req.lat), float(req.lon), req.observation_month
            )
    except LocationNotFound as exc:
        raise ServiceError(
            status_code=404,
            code="LOCATION_NOT_FOUND",
            message=str(exc),
            retryable=False,
        ) from exc
    except SpatialDataUnavailable as exc:
        raise ServiceError(
            status_code=503,
            code="SPATIAL_DATA_UNAVAILABLE",
            message="verified serving data is unavailable",
            retryable=True,
        ) from exc

    requested_targets = list(MODEL_TARGETS) if req.include_all_targets else list(req.targets or [])
    targets = resolve_targets(requested_targets, req.composite_features)
    predictions: dict[str, dict[str, Any]] = {}
    # Do not wrap with try/except ModelUnavailable, instead just skip silently
    for target in targets:
        try:
            predictions[target] = _predict_target(target, match.features)
        except ModelUnavailable:
            pass

    try:
        composites = CompositeFeaturesEngine.build_requested(
            req.composite_features, predictions, match.features
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ServiceError(
            status_code=500,
            code="COMPOSITE_CALCULATION_FAILED",
            message="verified model outputs could not produce the requested composite",
            retryable=False,
        ) from exc

    return {
        "api_version": API_VERSION,
        "contract_version": CONTRACT_VERSION,
        "catalog_version": model_catalog.catalog_version,
        "request_id": request_id,
        "status": "success",
        "location": _location_payload(match),
        "predictions": predictions,
        "composite_features": composites,
        "provenance": _provenance_payload(match),
        "execution_metadata": {
            "response_time_ms": 0.0,
            "queue_wait_ms": 0.0,
            "cached": False,
            "models_loaded_count": model_manager.diagnostics()["loaded_model_count"],
        },
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_indicators(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    started = time.perf_counter()
    request_id = payload.request_id or str(request.state.request_id)
    requested_targets = list(MODEL_TARGETS) if payload.include_all_targets else list(payload.targets or [])
    expanded_targets = resolve_targets(requested_targets, payload.composite_features)
    if len(expanded_targets) > MAX_EXPANDED_SYNC_TARGETS:
        raise ServiceError(
            status_code=413,
            code="REQUEST_TOO_EXPENSIVE",
            message=(
                f"synchronous inference is limited to {MAX_EXPANDED_SYNC_TARGETS} expanded "
                "targets; split the request or use a future async batch endpoint"
            ),
            retryable=False,
        )
    cache_payload = payload.model_dump(exclude={"request_id"})
    namespace = (
        f"prediction:{CONTRACT_VERSION}:{model_catalog.catalog_version}:"
        f"{model_catalog.feature_dataset_sha256[:16]}"
    )
    cache_key = cache_manager.generate_cache_key(namespace, cache_payload)
    cached = cache_manager.get(cache_key)
    if cached is not None:
        cached["request_id"] = request_id
        cached["execution_metadata"]["cached"] = True
        cached["execution_metadata"]["response_time_ms"] = round(
            (time.perf_counter() - started) * 1000, 2
        )
        cached["execution_metadata"]["queue_wait_ms"] = 0.0
        cached["execution_metadata"]["models_loaded_count"] = model_manager.diagnostics()[
            "loaded_model_count"
        ]
        return PredictionResponse.model_validate(cached)

    try:
        response_payload, queue_wait_ms = await request_queue.execute(
            _compute_prediction, payload, request_id
        )
    except QueueTimeout as exc:
        raise ServiceError(
            status_code=503,
            code="INFERENCE_CAPACITY_EXCEEDED",
            message="model server is busy; retry after a short delay",
            retryable=True,
        ) from exc
    except ExecutionTimeout as exc:
        raise ServiceError(
            status_code=504,
            code="INFERENCE_TIMEOUT",
            message="model inference exceeded the synchronous execution deadline",
            retryable=True,
        ) from exc

    response_payload["execution_metadata"]["queue_wait_ms"] = queue_wait_ms
    response_payload["execution_metadata"]["response_time_ms"] = round(
        (time.perf_counter() - started) * 1000, 2
    )
    validated = PredictionResponse.model_validate(response_payload)
    cache_manager.set(cache_key, validated.model_dump(mode="json"))
    return validated


@router.get("/models")
async def list_models() -> dict[str, Any]:
    if model_catalog.load_error:
        raise ServiceError(
            status_code=503,
            code="MODEL_CATALOG_UNAVAILABLE",
            message="the model catalog is unavailable",
            retryable=True,
        )
    try:
        model_catalog.verify_release()
    except CatalogError as exc:
        raise ServiceError(
            status_code=503,
            code="MODEL_CATALOG_UNAVAILABLE",
            message="the model release failed artifact integrity verification",
            retryable=True,
        ) from exc
    return {
        "api_version": API_VERSION,
        "contract_version": CONTRACT_VERSION,
        "catalog_version": model_catalog.catalog_version,
        "feature_dataset_sha256": model_catalog.feature_dataset_sha256,
        "spatial_index_sha256": model_catalog.spatial_index_sha256,
        "capabilities": {
            "max_expanded_sync_targets": MAX_EXPANDED_SYNC_TARGETS,
            "supports_composite_only_requests": True,
            "composite_dependencies": {
                name: list(dependencies)
                for name, dependencies in COMPOSITE_DEPENDENCIES.items()
            },
        },
        "models": model_catalog.list_models(),
    }


@router.get("/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/ready")
async def readiness() -> dict[str, Any]:
    try:
        model_catalog.verify_release()
    except CatalogError as exc:
        raise ServiceError(
            status_code=503,
            code="SERVICE_NOT_READY",
            message="required model or serving-data integrity verification failed",
            retryable=True,
        ) from exc
    catalog = model_catalog.readiness()
    spatial = spatial_manager.readiness()
    ready = (
        catalog["loaded"]
        and catalog["available_model_count"] == catalog["required_model_count"]
        and catalog["verified_model_count"] == catalog["required_model_count"]
        and catalog["serving_data_verified"]
        and spatial["loaded"]
        and spatial["feature_rows"] == spatial["spatial_rows"]
    )
    if not ready:
        raise ServiceError(
            status_code=503,
            code="SERVICE_NOT_READY",
            message="required model or spatial artifacts are unavailable",
            retryable=True,
        )
    return {
        "status": "ready",
        "catalog_version": model_catalog.catalog_version,
        "model_count": catalog["available_model_count"],
        "spatial_rows": spatial["spatial_rows"],
    }


@router.get("/health")
async def health_diagnostics() -> dict[str, Any]:
    return {
        "status": "healthy" if spatial_manager.is_loaded and not model_catalog.load_error else "degraded",
        "catalog": model_catalog.readiness(),
        "spatial": spatial_manager.readiness(),
        "model_cache": model_manager.diagnostics(),
        "request_queue": request_queue.get_metrics(),
        "cache": cache_manager.diagnostics(),
    }
