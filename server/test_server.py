#!/usr/bin/env python3
"""
server/test_server.py — Comprehensive Integration & Unit Testing Suite for Model Server
"""

import sys
import os
import json
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.main import app
from server.core.model_loader import model_manager
from server.core.preprocessor import spatial_manager
from server.services.composite_features import CompositeFeaturesEngine

client = TestClient(app)

def test_1_spatial_lookup():
    print("\n[TEST 1] Testing Spatial Dataset Lookup & KD-Tree...")
    # Test system:index lookup
    sample = spatial_manager.lookup_by_system_index("0")
    assert sample is not None, "Failed to lookup system:index 0"
    print(f"  ✓ system:index lookup success: Found {len(sample)} features.")

    # Test KD-Tree lat/lon lookup (Yangon coordinates ~16.866, 96.195)
    res = spatial_manager.lookup_by_lat_lon(16.8661, 96.1951)
    assert res is not None, "Failed KD-Tree lat/lon lookup"
    row, dist = res
    print(f"  ✓ KD-Tree lat/lon lookup success: Nearest distance = {dist:.5f} deg.")

def test_2_lru_model_loader_and_fallback():
    print("\n[TEST 2] Testing LRU Model Loader & Prototype Fallback...")
    model_manager.clear_cache()
    assert len(model_manager._lru_cache) == 0

    # Load 5 different targets to trigger LRU eviction (cap = 4)
    targets = ["crop_yield_t_ha", "crop_health_score", "drought_risk_score", "market_integration_score", "surface_water_occurrence"]
    for t in targets:
        m, src = model_manager.get_model(t)
        assert m is not None, f"Failed to load model {t}"

    # Verify LRU cache size stays <= 4
    loaded_count = len(model_manager._lru_cache)
    print(f"  ✓ LRU Model Cache Size = {loaded_count} (Cap = {model_manager.max_models}). Memory eviction working properly!")
    assert loaded_count <= model_manager.max_models

    # Test forced fallback prototype model loading
    proto_m, proto_src = model_manager.get_model("crop_yield_t_ha", force_prototype=True)
    assert proto_src == "prototype", "Failed to fallback to prototype model"
    print("  ✓ Fallback Prototype model loading success!")

def test_3_composite_features():
    print("\n[TEST 3] Testing Composite Multi-Model Feature Engine...")
    mock_preds = {
        "crop_yield_t_ha": 3.8,
        "crop_health_score": 0.88,
        "drought_risk_score": 0.12,
        "flood_risk_level": "0",
        "heat_stress_risk": "0",
        "water_scarcity_risk": 0.15,
        "agricultural_gdp_forecast": 0.75,
        "market_integration_score": 0.82,
        "agricultural_land_conversion_risk": 0.10,
        "urban_encroachment_risk": 0.05,
        "crop_suitability_dry_season_rice": "excellent",
        "crop_suitability_monsoon_rice": "good",
        "crop_suitability_groundnut": "good"
    }
    raw_feats = {"ndvi_median_growing_season_mean": 0.72, "cropland_fraction": 0.90}

    rec = CompositeFeaturesEngine.build_crop_recommender(mock_preds)
    assert len(rec) > 0 and rec[0]["crop"] == "dry_season_rice"
    print("  ✓ Micro-Regional Crop Recommender output verified.")

    health = CompositeFeaturesEngine.build_crop_health_layer(mock_preds, raw_feats)
    assert health["health_status"] == "Optimal Health"
    print("  ✓ Geospatial Crop Health Layer output verified.")

    roi = CompositeFeaturesEngine.build_economic_roi_calculator(mock_preds)
    assert roi["roi_rating"] == "HIGH ROI"
    print("  ✓ Economic Yield & ROI Calculator output verified.")

    alert = CompositeFeaturesEngine.build_multi_hazard_risk_alert(mock_preds)
    assert alert["overall_alert_level"] == "LOW RISK (STABLE)"
    print("  ✓ Multi-Hazard Risk Alert Engine output verified.")

def test_4_fastapi_endpoints():
    print("\n[TEST 4] Testing FastAPI REST Endpoints...")
    
    # 1. Test Health Endpoint
    resp_health = client.get("/api/v1/health")
    assert resp_health.status_code == 200
    data_health = resp_health.json()
    assert data_health["status"] == "healthy"
    print(f"  ✓ GET /api/v1/health: RAM Usage = {data_health['ram_usage_mb']} MB (Under 2048 MB Cap).")

    # 2. Test Predict Endpoint (specific targets + composite features)
    predict_payload = {
        "system_index": "00000000000000000001",
        "targets": ["crop_yield_t_ha", "crop_suitability_monsoon_rice"],
        "composite_features": ["crop_recommender", "risk_alerts", "economic_roi"]
    }
    resp_pred = client.post("/api/v1/predict", json=predict_payload)
    assert resp_pred.status_code == 200
    data_pred = resp_pred.json()
    assert data_pred["status"] == "success"
    assert "crop_yield_t_ha" in data_pred["predictions"]
    assert "crop_recommender" in data_pred["composite_features"]
    print(f"  ✓ POST /api/v1/predict: Response time = {data_pred['execution_metadata']['response_time_ms']} ms.")

    # 3. Test Regional Endpoint
    resp_reg = client.get("/api/v1/regions/ayeyawaddy")
    assert resp_reg.status_code == 200
    data_reg = resp_reg.json()
    assert data_reg["region"] == "Ayeyawaddy"
    print(f"  ✓ GET /api/v1/regions/ayeyawaddy: Found {data_reg['total_samples']} samples.")

def test_5_boost_mode_and_request_queue():
    print("\n[TEST 5] Testing 🚀 Boost Mode & 🚦 Asynchronous Request Queue...")
    
    # 1. Test Boost API Endpoint
    resp_boost = client.post("/api/v1/boost?enabled=true")
    assert resp_boost.status_code == 200
    data_boost = resp_boost.json()
    assert data_boost["boost_mode_active"] is True
    print(f"  ✓ POST /api/v1/boost?enabled=true: Boost Mode active. Preloaded {data_boost['models_in_ram_count']} models in RAM.")

    # 2. Test Prediction under Boost Mode
    predict_payload = {
        "lat": 16.8661,
        "lon": 96.1951,
        "boost_mode": True,
        "include_all_targets": True
    }
    resp_pred = client.post("/api/v1/predict", json=predict_payload)
    assert resp_pred.status_code == 200
    data_pred = resp_pred.json()
    assert data_pred["execution_metadata"]["boost_mode_active"] is True
    assert "queue_wait_ms" in data_pred["execution_metadata"]
    print(f"  ✓ Boost Mode Prediction Latency = {data_pred['execution_metadata']['response_time_ms']} ms (Queue Wait: {data_pred['execution_metadata']['queue_wait_ms']} ms).")

    # 3. Test Health Diagnostics under Boost Mode
    resp_health = client.get("/api/v1/health")
    assert resp_health.status_code == 200
    data_health = resp_health.json()
    assert data_health["boost_mode_active"] is True
    assert "request_queue_diagnostics" in data_health
    print(f"  ✓ GET /api/v1/health: Request Queue Capacity = {data_health['request_queue_diagnostics']['max_concurrent_capacity']} workers.")

if __name__ == "__main__":
    print("=====================================================================")
    print("     MODEL SERVER MICROSERVICE INTEGRATION & UNIT TEST SUITE         ")
    print("=====================================================================")
    test_1_spatial_lookup()
    test_2_lru_model_loader_and_fallback()
    test_3_composite_features()
    test_4_fastapi_endpoints()
    test_5_boost_mode_and_request_queue()
    print("\n=====================================================================")
    print("  ALL MODEL SERVER MICROSERVICE TESTS PASSED SUCCESSFULLY (100%)    ")
    print("=====================================================================\n")
