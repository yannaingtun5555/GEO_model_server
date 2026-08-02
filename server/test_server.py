"""Integration tests for the strict, fail-closed serving contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pytest
from scipy.spatial import cKDTree
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

from server.config import CROPS
from server.core.catalog import model_catalog
from server.core.model_loader import LRUModelManager, ModelUnavailable, model_manager
from server.core.preprocessor import spatial_manager
from server.core.preprocessor import _haversine_km, _unit_sphere_coordinates
from server.core.request_queue import AsyncRequestQueue, ExecutionTimeout, QueueTimeout
from server.main import app
from server.services.composite_features import CompositeFeaturesEngine, resolve_targets


PRIMARY_MODELS_AVAILABLE = all(
    model_catalog.artifact_path(target).is_file() for target in model_catalog.models
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def valid_request() -> dict:
    return {
        "lat": 15.731919,
        "lon": 95.324433,
        "observation_month": "2018-01",
        "targets": ["heat_stress_risk"],
    }


def test_serving_artifacts_are_row_aligned() -> None:
    status = spatial_manager.readiness()
    assert status["loaded"] is True
    assert status["feature_rows"] == status["spatial_rows"] == 1_029_348
    assert status["available_month_start"] == "2018-01"
    assert spatial_manager.features.memory_usage(deep=True).sum() < 400 * 1024 * 1024
    assert spatial_manager.spatial_index.memory_usage(deep=True).sum() < 100 * 1024 * 1024


def test_nearest_lookup_uses_spherical_geometry() -> None:
    query = np.asarray([[28.0, 96.0]])
    candidates = np.asarray([[28.09, 96.0], [28.0, 96.10]])
    raw_degree_choice = int(cKDTree(candidates).query(query[0])[1])
    spherical_choice = int(
        cKDTree(_unit_sphere_coordinates(candidates)).query(
            _unit_sphere_coordinates(query)[0]
        )[1]
    )
    distances = [
        _haversine_km(query[0, 0], query[0, 1], point[0], point[1])
        for point in candidates
    ]
    assert raw_degree_choice == 0
    assert spherical_choice == int(np.argmin(distances)) == 1


def test_catalog_declares_all_checksum_backed_models() -> None:
    status = model_catalog.readiness()
    assert status["loaded"] is True
    assert status["declared_model_count"] == status["required_model_count"] == 40
    assert status["available_model_count"] == (40 if PRIMARY_MODELS_AVAILABLE else 0)
    assert len(model_catalog.catalog_version) == 64


def test_liveness_readiness_and_catalog(client: TestClient) -> None:
    assert client.get("/api/v1/live").json() == {"status": "alive"}
    ready = client.get("/api/v1/ready")
    assert ready.status_code == (200 if PRIMARY_MODELS_AVAILABLE else 503)
    if PRIMARY_MODELS_AVAILABLE:
        assert ready.json()["model_count"] == 40
    catalog = client.get("/api/v1/models")
    assert catalog.status_code == (200 if PRIMARY_MODELS_AVAILABLE else 503)
    if not PRIMARY_MODELS_AVAILABLE:
        assert catalog.json()["error"]["code"] == "MODEL_CATALOG_UNAVAILABLE"
        return
    assert catalog.json()["feature_dataset_sha256"] == model_catalog.feature_dataset_sha256
    assert catalog.json()["spatial_index_sha256"] == model_catalog.spatial_index_sha256
    capabilities = catalog.json()["capabilities"]
    assert capabilities["supports_composite_only_requests"] is True
    assert capabilities["max_expanded_sync_targets"] >= 17
    assert capabilities["composite_dependencies"]["economic_roi"] == []
    assert len(capabilities["composite_dependencies"]["crop_recommender"]) == 17
    models = catalog.json()["models"]
    assert len(models) == 40
    assert all(model["deployment_status"] == "experimental" for model in models)
    assert all(model["field_validated"] is False for model in models)
    assert sum(model["validation_status"] == "flagged" for model in models) >= 20


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"lat": 16.8, "lon": 96.1, "targets": ["crop_health_score"]},
        {
            "sample_id": "x",
            "lat": 16.8,
            "lon": 96.1,
            "observation_month": "2018-01",
            "targets": ["crop_health_score"],
        },
        {
            "lat": 16.8,
            "lon": 96.1,
            "observation_month": "2018-13",
            "targets": ["crop_health_score"],
        },
        {
            "lat": 16.8,
            "lon": 96.1,
            "observation_month": "2018-01",
            "targets": ["not_a_model"],
        },
    ],
)
def test_invalid_requests_fail_validation(client: TestClient, payload: dict) -> None:
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"


def test_out_of_release_location_never_uses_a_synthetic_row(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        json={
            "lat": 28.0,
            "lon": 92.1,
            "observation_month": "2018-01",
            "targets": ["heat_stress_risk"],
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "LOCATION_NOT_FOUND"


@pytest.mark.skipif(not PRIMARY_MODELS_AVAILABLE, reason="primary artifacts are mounted at deploy time")
def test_primary_model_prediction_has_provenance_and_confidence(client: TestClient) -> None:
    response = client.post("/api/v1/predict", json=valid_request())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["contract_version"] == "model-inference-v1"
    assert body["catalog_version"] == model_catalog.catalog_version
    assert body["location"]["sample_id"] == "mm_1839_396__2018-01"
    assert body["location"]["region"] == "ayeyawaddy"
    assert body["location"]["distance_km"] < 0.001
    prediction = body["predictions"]["heat_stress_risk"]
    assert prediction["model_source"] == "primary"
    assert prediction["confidence"] is not None
    assert prediction["confidence_kind"] == "random_forest_vote_share_uncalibrated"
    assert len(prediction["artifact_sha256"]) == 64
    assert prediction["deployment_status"] == "experimental"
    assert body["provenance"]["field_validated"] is False
    assert body["provenance"]["label_source"] == "rule_engineered_surrogate"


@pytest.mark.skipif(not PRIMARY_MODELS_AVAILABLE, reason="primary artifacts are mounted at deploy time")
def test_identical_request_is_versioned_and_cached(client: TestClient) -> None:
    first = client.post("/api/v1/predict", json=valid_request())
    second = client.post("/api/v1/predict", json=valid_request())
    assert first.status_code == second.status_code == 200
    assert second.json()["execution_metadata"]["cached"] is True
    assert second.json()["execution_metadata"]["queue_wait_ms"] == 0.0
    assert first.json()["predictions"] == second.json()["predictions"]


@pytest.mark.skipif(not PRIMARY_MODELS_AVAILABLE, reason="primary artifacts are mounted at deploy time")
def test_released_forest_is_not_pruned_at_runtime(client: TestClient) -> None:
    client.post("/api/v1/predict", json=valid_request())
    artifact, _ = model_manager.get_model("heat_stress_risk")
    assert artifact["model"].n_estimators == 500
    assert len(artifact["model"].estimators_) == 500


def test_memory_dos_boost_route_was_removed(client: TestClient) -> None:
    assert client.post("/api/v1/boost?enabled=true").status_code == 404


def test_manifest_is_json_and_tracks_spatial_checksum() -> None:
    manifest_path = Path(__file__).resolve().parent.parent / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["spatial_index"]["sha256"] == model_catalog.spatial_index_sha256
    assert manifest["governance"]["production_approval"] is False
    assert manifest["feature_dataset"]["filename"] == "features_serving.parquet"


def test_timed_out_worker_keeps_its_capacity_slot() -> None:
    async def scenario() -> None:
        blocker = threading.Event()
        queue = AsyncRequestQueue(
            max_concurrent=1,
            queue_timeout=0.02,
            execution_timeout=0.02,
        )
        with pytest.raises(ExecutionTimeout):
            await queue.execute(blocker.wait)
        assert queue.active_requests == 1
        with pytest.raises(QueueTimeout):
            await queue.execute(lambda: None)
        blocker.set()
        for _ in range(100):
            if queue.active_requests == 0:
                break
            await asyncio.sleep(0.005)
        assert queue.active_requests == 0

    asyncio.run(scenario())


def test_unavailable_roi_has_no_inference_dependencies() -> None:
    assert resolve_targets([], ["economic_roi"]) == []
    roi = CompositeFeaturesEngine.build_economic_roi_calculator({})
    assert roi["status"] == "unavailable"


def test_composite_only_roi_request_returns_typed_unavailable(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        json={
            "lat": 15.731919,
            "lon": 95.324433,
            "observation_month": "2018-01",
            "composite_features": ["economic_roi"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["predictions"] == {}
    assert body["composite_features"]["economic_roi"]["status"] == "unavailable"


def test_crop_recommender_returns_tiers_not_false_cross_model_rank() -> None:
    predictions = {}
    for index, crop in enumerate(CROPS):
        label = "excellent" if index < 2 else "good"
        predictions[f"crop_suitability_{crop}"] = {
            "value": label,
            "confidence": 0.99 - index / 100,
        }
    result = CompositeFeaturesEngine.build_crop_recommender(predictions)
    assert result["strict_ranking_available"] is False
    assert result["probability_calibrated"] is False
    assert result["top_suitability_tier"] == "excellent"
    assert len(result["top_recommendations"]) == 2
    assert result["reason_code"] == "CROSS_MODEL_CALIBRATION_REQUIRED"


def test_unknown_risk_class_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported risk class"):
        CompositeFeaturesEngine.build_multi_hazard_risk_alert(
            {
                "flood_risk_level": {"value": "invented"},
                "drought_risk_score": {"value": 0.2},
                "heat_stress_risk": {"value": 0},
                "soil_erosion_risk": {"value": 0},
                "water_scarcity_risk": {"value": 0.2},
            }
        )


def test_ci_fixture_deserializes_and_detects_checksum_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = RandomForestRegressor(n_estimators=2, random_state=7).fit(
        np.asarray([[0.0], [0.5], [1.0]]),
        np.asarray([0.1, 0.5, 0.9]),
    )
    fixture_path = tmp_path / "fixture.pkl"
    joblib.dump({"model": model, "features": ["x"]}, fixture_path)
    expected_digest = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    schema_digest = hashlib.sha256(b'["x"]').hexdigest()

    class FixtureCatalog:
        metadata = {
            "artifact_size_bytes": fixture_path.stat().st_size,
            "artifact_sha256": expected_digest,
            "input_schema_sha256": schema_digest,
        }

        def get_model(self, target: str) -> dict:
            assert target == "crop_health_score"
            return self.metadata

        def artifact_path(self, target: str) -> Path:
            assert target == "crop_health_score"
            return fixture_path

        def verify_model(self, target: str) -> None:
            assert target == "crop_health_score"
            actual = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
            if actual != self.metadata["artifact_sha256"]:
                from server.core.catalog import CatalogError

                raise CatalogError("fixture checksum mismatch")

    import server.core.model_loader as loader_module

    monkeypatch.setattr(loader_module, "model_catalog", FixtureCatalog())
    manager = LRUModelManager(max_models=1, max_ram_mb=4096)
    loaded, _ = manager.get_model("crop_health_score")
    assert 0.0 <= float(loaded["model"].predict([[0.25]])[0]) <= 1.0

    manager.clear_cache()
    fixture_path.write_bytes(fixture_path.read_bytes() + b"corrupt")
    with pytest.raises(ModelUnavailable, match="checksum mismatch"):
        manager.get_model("crop_health_score")
