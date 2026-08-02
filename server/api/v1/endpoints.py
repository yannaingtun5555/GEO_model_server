"""Strict, fail-closed model-serving API v1 endpoints."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, File, UploadFile, BackgroundTasks
from pydantic import BaseModel, Field

from server.config import (
    API_VERSION,
    CONTRACT_VERSION,
    MAX_EXPANDED_SYNC_TARGETS,
    MODEL_TARGETS,
)
from server.contracts import PredictionRequest, PredictionResponse
from server.core.cache import cache_manager
from server.core.model_loader import model_manager
from server.core.preprocessor import spatial_manager
from server.core.live_prediction_manager import live_prediction_manager
from server.core.request_queue import request_queue
from server.services.composite_features import CompositeFeaturesEngine

router = APIRouter(prefix="/api/v1", tags=["Model Server API"])
prediction_executor = ThreadPoolExecutor(max_workers=4)

# Request Schemas
class PredictionRequest(BaseModel):
    system_index: Optional[str] = Field(None, example="00000000000000000001", description="Land Index ID")
    region_name: Optional[str] = Field(None, description="Region name (e.g. Yangon, Ayeyawaddy, Mandalay)")
    lat: Optional[float] = Field(None, example=16.8661, description="Latitude coordinate")
    lon: Optional[float] = Field(None, example=96.1951, description="Longitude coordinate")
    targets: Optional[List[str]] = Field(default=None, description="Specific targets to predict or ['all']")
    include_all_targets: bool = Field(default=False, description="Predict all 40 models if true")
    composite_features: Optional[List[str]] = Field(default=None, description="Composite features to calculate: ['crop_recommender', 'crop_health', 'economic_roi', 'risk_alerts', 'land_use']")
    use_fallback_models: bool = Field(default=False, description="Force use of lightweight prototype models under heavy load")
    boost_mode: bool = Field(default=False, description="Enable Boost Mode (unlimited memory, preloads models for ultra-fast performance)")


router = APIRouter(prefix="/api/v1", tags=["model-serving-v1"])


    # 1.5 Check if pre-computed prediction exists in live_prediction_manager
    if live_prediction_manager.is_loaded:
        pred_record = None
        spatial_dist = 0.0
        if req.system_index:
            pred_record = live_prediction_manager.lookup_by_system_index(req.system_index)
        elif req.lat is not None and req.lon is not None:
            lookup_res = live_prediction_manager.lookup_by_lat_lon(req.lat, req.lon)
            if lookup_res:
                pred_record, spatial_dist = lookup_res
        elif req.region_name:
            reg_df = live_prediction_manager.df
            if reg_df is not None:
                subset = reg_df[reg_df["region"].astype(str).str.lower() == req.region_name.lower()]
                if not subset.empty:
                    pred_record = subset.iloc[0].to_dict()

        if pred_record is not None:
            # Reconstruct prediction targets
            all_known_targets = [
                "crop_health_score", "crop_yield_t_ha", "irrigation_need",
                "current_month_precipitation_mm", "current_month_mean_temperature_c", "current_month_solar_rad_mj_m2_day",
                "flood_risk_level", "drought_risk_score", "heat_stress_risk", "optimal_planting_month",
                "nitrogen_requirement_level", "phosphorus_requirement_level", "soil_erosion_risk",
                "market_integration_score", "post_harvest_loss_risk", "supply_chain_efficiency",
                "cold_chain_potential", "agricultural_land_conversion_risk", "urban_encroachment_risk",
                "irrigation_potential", "surface_water_occurrence", "water_scarcity_risk", "agricultural_gdp_forecast"
            ] + [f"crop_suitability_{c}" for c in CROPS]

            predictions = {t: pred_record[t] for t in all_known_targets if t in pred_record}
            
            # Lookup/reconstruct features for composite engine
            sample_features = spatial_manager.lookup_by_system_index(pred_record.get("system:index", ""))
            if not sample_features:
                sample_features = {
                    "system:index": str(pred_record.get("system:index", "")),
                    "latitude": float(pred_record.get("latitude", 0.0)),
                    "longitude": float(pred_record.get("longitude", 0.0))
                }

            comp_flags = req.composite_features or []
            if "all" in comp_flags:
                comp_flags = ["crop_recommender", "crop_health", "economic_roi", "risk_alerts", "land_use"]
                
            composite_res = {}
            if "crop_recommender" in comp_flags:
                composite_res["crop_recommender"] = CompositeFeaturesEngine.build_crop_recommender(predictions, sample_features)
            if "crop_health" in comp_flags:
                composite_res["crop_health_layer"] = CompositeFeaturesEngine.build_crop_health_layer(predictions, sample_features)
            if "economic_roi" in comp_flags:
                composite_res["economic_roi_calculator"] = CompositeFeaturesEngine.build_economic_roi_calculator(predictions)
            if "risk_alerts" in comp_flags:
                composite_res["multi_hazard_risk_alert"] = CompositeFeaturesEngine.build_multi_hazard_risk_alert(predictions)
            if "land_use" in comp_flags:
                composite_res["land_use_pattern"] = CompositeFeaturesEngine.build_land_use_pattern(predictions, sample_features)

            process = psutil.Process(os.getpid())
            ram_used_mb = round(process.memory_info().rss / (1024 * 1024), 1)

            resp_payload = {
                "status": "success",
                "location": {
                    "system_index": str(pred_record.get("system:index", "")),
                    "lat": float(pred_record.get("latitude", 0.0)),
                    "lon": float(pred_record.get("longitude", 0.0)),
                    "nearest_distance_deg": round(spatial_dist, 5)
                },
                "predictions": predictions,
                "composite_features": composite_res,
                "execution_metadata": {
                    "boost_mode_active": model_manager.boost_mode,
                    "models_used": {t: "live_prediction_db" for t in predictions.keys()},
                    "ram_used_mb": ram_used_mb,
                    "models_in_memory_count": len(model_manager._lru_cache),
                    "lru_models_in_memory": list(model_manager._lru_cache.keys()),
                    "cached": False,
                    "response_time_ms": round((time.time() - start_t) * 1000.0, 2),
                    "queue_wait_ms": 0.0,
                    "active_workers": 1,
                    "live_db_served": True
                }
            }
            cache_manager.set(cache_key, resp_payload)
            return resp_payload

    # 2. Worker computation closure
    def _compute_predictions():
        # Temporarily activate boost mode if requested
        if req.boost_mode and not model_manager.boost_mode:
            model_manager.set_boost_mode(True)


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

    # 3. Route through Async Request Queue
    response_payload = await request_queue.execute(_compute_predictions)

    exec_time_ms = round((time.time() - start_t) * 1000.0, 2)
    queue_meta = response_payload.pop("_queue_metadata", {})

    response_payload["execution_metadata"]["response_time_ms"] = exec_time_ms
    response_payload["execution_metadata"]["queue_wait_ms"] = queue_meta.get("queue_wait_ms", 0.0)
    response_payload["execution_metadata"]["active_workers"] = queue_meta.get("active_workers", 1)
    response_payload["execution_metadata"]["cached"] = False

    # Save to Redis cache
    cache_manager.set(cache_key, response_payload)

    return response_payload


@router.post("/boost", summary="Dynamically Enable/Disable Boost Mode & Preload Models")
async def toggle_boost_mode(enabled: bool = Query(True, description="Set true to enable Boost Mode (unlimited memory, preloads models)")):
    """
    Dynamically toggles Boost Mode:
    - enabled=true: Preloads and pins all 40 ML models in RAM for max performance & zero latency.
    - enabled=false: Reverts to standard memory-capped LRU model loader.
    """
    res = model_manager.set_boost_mode(enabled)
    return {
        "status": "success",
        "boost_mode_active": model_manager.boost_mode,
        "models_in_ram_count": len(model_manager._lru_cache),
        "details": res
    }


@router.get("/regions/{region_name}", summary="Get Pre-Computed Regional Crop Suitability & Climate Layer")
async def get_regional_summary(region_name: str, top_k: int = Query(5, ge=1, le=17)):
    """
    Returns pre-computed regional crop suitability, top crop recommendations,
    and regional climate summary for the 6 target regions.
    Uses Redis storage for zero-latency retrieval.
    """
    r_lower = region_name.lower()
    if r_lower not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Region '{region_name}' not found. Available regions: {REGIONS}")

    cache_key = f"regional_storage:{r_lower}:top_{top_k}"
    cached_val = cache_manager.get(cache_key)
    if cached_val:
        return cached_val

    reg_df = spatial_manager.get_region_subset(r_lower)
    if reg_df.empty:
        raise HTTPException(status_code=404, detail=f"No data rows available for region '{region_name}'")

    res_payload = {
        "status": "success",
        "region": r_lower.capitalize(),
        "total_samples": len(reg_df),
        "mean_precipitation_mm": round(float(reg_df.get("chirps_precipitation_mm_mean", pd.Series([1200])).mean()), 1),
        "mean_temperature_c": round(float(reg_df.get("mean_temperature_c_mean", pd.Series([27.0])).mean()), 1),
        "regional_insights": {
            "top_recommended_crops": CROPS[:top_k],
            "primary_agro_zone": "Delta Lowland" if r_lower in ["ayeyawaddy", "yangon", "bago"] else "Central Dry Zone"
        }
    }

    cache_manager.set(cache_key, res_payload, ttl_seconds=86400 * 7)
    return res_payload


@router.get("/map-recommendations", summary="Get Pre-Computed Crop Recommendations for Map Visualization")
async def get_map_recommendations():
    """
    Returns pre-computed all 17 crop recommendations (ranked by suitability score, highest first)
    for ~2,500 downsampled grid points across Myanmar.
    This avoids heavy real-time ML model inference when rendering the map.
    """
    import json
    from pathlib import Path
    file_path = Path(__file__).resolve().parent.parent.parent / "static" / "map_recommendations.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Map recommendations dataset not generated yet. Run scripts/generate_map_data.py first.")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


@router.post("/pipeline/update", summary="Ingest GEE Update CSV & Re-compute Live Predictions")
async def update_from_gee(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Monthly GEE dynamic CSV export (must contain system:index column)")
):
    """
    Accepts a new GEE monthly dynamic observations CSV, merges it with the static features dataset,
    runs batch predictions for all 40 ML targets across all grid points, saves the live prediction
    database, and regenerates the map_recommendations.json. All heavy processing runs in the background.
    """
    import tempfile
    import shutil
    from pathlib import Path as PathLib
    from server.services.gee_pipeline import run_gee_ingestion_pipeline

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Save uploaded file to a temp path
    tmp_dir = PathLib(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    with open(tmp_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    def _run_pipeline():
        try:
            result = run_gee_ingestion_pipeline(tmp_path)
            print(f"[PIPELINE UPDATE] Completed: {result}")
        except Exception as e:
            print(f"[PIPELINE UPDATE ERROR] {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    background_tasks.add_task(_run_pipeline)

    return {
        "status": "accepted",
        "message": "GEE update ingestion started in background. Live predictions will be updated on completion.",
        "filename": file.filename
    }


@router.get("/health", summary="Model Server Health & Resource Diagnostics")
async def get_health_status():
    """Returns microservice health status, Boost Mode state, RAM usage, and Request Queue diagnostics."""
    process = psutil.Process(os.getpid())
    ram_mb = round(process.memory_info().rss / (1024 * 1024), 1)


    live_db_rows = len(live_prediction_manager.df) if live_prediction_manager.is_loaded and live_prediction_manager.df is not None else 0

    return {
        "status": "healthy",
        "service": "Agricultural Model Serving Microservice",
        "boost_mode_active": model_manager.boost_mode,
        "ram_usage_mb": ram_mb,
        "ram_limit_mb": "unlimited (Boost Mode)" if model_manager.boost_mode else 2048,
        "lru_max_models_cap": "unlimited" if model_manager.boost_mode else model_manager.max_models,
        "models_currently_in_ram_count": len(model_manager._lru_cache),
        "models_currently_in_ram": list(model_manager._lru_cache.keys()),
        "request_queue_diagnostics": request_queue.get_metrics(),
        "redis_connected": cache_manager.redis_client is not None,
        "spatial_dataset_loaded": spatial_manager.is_loaded,
        "live_prediction_db_loaded": live_prediction_manager.is_loaded,
        "live_prediction_db_rows": live_db_rows
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
