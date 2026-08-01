#!/usr/bin/env python3
"""
server/api/v1/endpoints.py — REST API Endpoint Handlers for FastAPI
"""

import time
import os
import psutil
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from server.config import REGIONS, CROPS
from server.core.cache import cache_manager
from server.core.model_loader import model_manager
from server.core.preprocessor import spatial_manager
from server.services.composite_features import CompositeFeaturesEngine

router = APIRouter(prefix="/api/v1", tags=["Model Server API"])

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


@router.post("/predict", summary="Predict Agricultural Indicators & Crop Suitability")
async def predict_indicators(req: PredictionRequest):
    """
    Primary Model Server Endpoint:
    Lookup land sample by `system_index` OR `(lat, lon)` coordinates, run predictions
    for requested targets, calculate composite intelligence features, and return payload.
    """
    start_t = time.time()

    # 1. Generate Cache Key
    cache_payload = req.dict()
    cache_key = cache_manager.generate_cache_key("predict", cache_payload)
    
    cached_resp = cache_manager.get(cache_key)
    if cached_resp:
        cached_resp["execution_metadata"]["cached"] = True
        cached_resp["execution_metadata"]["response_time_ms"] = round((time.time() - start_t) * 1000.0, 2)
        return cached_resp

    # 2. Spatial Data Lookup
    sample_features = None
    spatial_dist = 0.0

    if req.region_name:
        sample_features = spatial_manager.lookup_by_region(req.region_name)
    elif req.system_index:
        sample_features = spatial_manager.lookup_by_system_index(req.system_index)
    
    if sample_features is None and req.lat is not None and req.lon is not None:
        lookup_res = spatial_manager.lookup_by_lat_lon(req.lat, req.lon)
        if lookup_res:
            sample_features, spatial_dist = lookup_res

    if sample_features is None:
        # If no spatial dataset row found, construct default fallback sample
        sample_features = {
            "system:index": req.system_index or "custom_location",
            "latitude": req.lat or 16.8661,
            "longitude": req.lon or 96.1951,
            "elevation_m": 15.0,
            "mean_temperature_c_mean": 27.5,
            "chirps_precipitation_mm_mean": 1800.0,
            "solar_radiation_mj_m2_day_mean": 18.5,
            "ndvi_median_mean": 0.65,
            "cropland_fraction": 0.85
        }

    # 3. Determine Targets to Predict
    all_known_targets = [
        "crop_health_score", "crop_yield_t_ha", "irrigation_need",
        "current_month_precipitation_mm", "current_month_mean_temperature_c", "current_month_solar_rad_mj_m2_day",
        "flood_risk_level", "drought_risk_score", "heat_stress_risk", "optimal_planting_month",
        "nitrogen_requirement_level", "phosphorus_requirement_level", "soil_erosion_risk",
        "market_integration_score", "post_harvest_loss_risk", "supply_chain_efficiency",
        "cold_chain_potential", "agricultural_land_conversion_risk", "urban_encroachment_risk",
        "irrigation_potential", "surface_water_occurrence", "water_scarcity_risk", "agricultural_gdp_forecast"
    ] + [f"crop_suitability_{c}" for c in CROPS]

    comp_flags = req.composite_features or []
    if "all" in comp_flags:
        comp_flags = ["crop_recommender", "crop_health", "economic_roi", "risk_alerts", "land_use"]

    crop_suit_targets = [f"crop_suitability_{c}" for c in CROPS]

    if req.include_all_targets or (req.targets and "all" in req.targets):
        target_list = all_known_targets
    elif "crop_recommender" in comp_flags:
        # Automatically include all 17 crop suitability targets for proper ranking
        req_targets = req.targets or []
        target_list = list(set(req_targets + crop_suit_targets + ["crop_yield_t_ha", "crop_health_score"]))
    elif req.targets:
        target_list = [t for t in req.targets if t != "all"]
    else:
        # Default essential targets if none specified
        target_list = ["crop_yield_t_ha", "crop_health_score", "drought_risk_score"] + crop_suit_targets

    # 4. Predict Targets via LRU Model Manager
    predictions = {}
    models_used = {}

    for target in target_list:
        model_artifact, source_type = model_manager.get_model(target, force_prototype=req.use_fallback_models)
        if model_artifact and "model" in model_artifact:
            model = model_artifact["model"]
            feats = model_artifact.get("features", [])
            le = model_artifact.get("label_encoder")

            # Extract inputs
            x_vals = [float(sample_features.get(f, 0.0) or 0.0) for f in feats]
            import pandas as pd
            X_in = pd.DataFrame([x_vals], columns=feats)

            try:
                pred_raw = model.predict(X_in)[0]
                if le is not None:
                    if isinstance(pred_raw, (int, float, str)) and hasattr(le, "classes_"):
                        try:
                            pred_val = str(le.classes_[int(pred_raw)])
                        except Exception:
                            pred_val = str(pred_raw)
                    else:
                        pred_val = str(pred_raw)
                else:
                    pred_val = float(pred_raw) if isinstance(pred_raw, (float, int)) else str(pred_raw)

                predictions[target] = pred_val
                models_used[target] = source_type
            except Exception as e:
                predictions[target] = f"Error: {e}"
                models_used[target] = "error"

    # 5. Composite Feature Calculations
    composite_res = {}
    comp_flags = req.composite_features or []
    if "all" in comp_flags:
        comp_flags = ["crop_recommender", "crop_health", "economic_roi", "risk_alerts", "land_use"]

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

    # RAM Usage calculation
    process = psutil.Process(os.getpid())
    ram_used_mb = round(process.memory_info().rss / (1024 * 1024), 1)
    exec_time_ms = round((time.time() - start_t) * 1000.0, 2)

    response_payload = {
        "status": "success",
        "location": {
            "system_index": str(sample_features.get("system:index", req.system_index)),
            "lat": float(sample_features.get("latitude", req.lat or 0.0)),
            "lon": float(sample_features.get("longitude", req.lon or 0.0)),
            "nearest_distance_deg": round(spatial_dist, 5)
        },
        "predictions": predictions,
        "composite_features": composite_res,
        "execution_metadata": {
            "response_time_ms": exec_time_ms,
            "cached": False,
            "models_used": models_used,
            "ram_used_mb": ram_used_mb,
            "lru_models_in_memory": list(model_manager._lru_cache.keys())
        }
    }

    # Save to Redis cache
    cache_manager.set(cache_key, response_payload)

    return response_payload


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

    # Load regional recommendations from CSV or spatial dataset
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

    cache_manager.set(cache_key, res_payload, ttl_seconds=86400 * 7)   # Cache for 7 days
    return res_payload


@router.get("/health", summary="Model Server Health & Resource Diagnostics")
async def get_health_status():
    """Returns microservice health status, memory usage (RAM cap 2GB), and loaded LRU models."""
    process = psutil.Process(os.getpid())
    ram_mb = round(process.memory_info().rss / (1024 * 1024), 1)

    return {
        "status": "healthy",
        "service": "Agricultural Model Serving Microservice",
        "ram_usage_mb": ram_mb,
        "ram_limit_mb": 2048,
        "lru_max_models_cap": model_manager.max_models,
        "lru_models_currently_in_ram": list(model_manager._lru_cache.keys()),
        "redis_connected": cache_manager.redis_client is not None,
        "spatial_dataset_loaded": spatial_manager.is_loaded
    }
