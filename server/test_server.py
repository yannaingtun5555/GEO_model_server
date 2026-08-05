"""Integration tests for the CSV dataset pipeline endpoint."""

from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient

from server.config import MODEL_TARGETS
from server.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_liveness_readiness_and_models(client: TestClient) -> None:
    assert client.get("/api/v1/live").json() == {"status": "alive"}
    
    ready = client.get("/api/v1/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["model_targets_count"] == 40

    models = client.get("/api/v1/models")
    assert models.status_code == 200
    assert models.json()["total_targets"] == 40
    assert len(models.json()["targets"]) == 40


def test_health_diagnostics(client: TestClient) -> None:
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["available_targets_count"] == 40


def test_pipeline_run_with_csv(client: TestClient) -> None:
    csv_content = (
        "system:index,latitude,longitude,ndvi_median_mean,soil_soc_g_kg_0_30cm,soil_ph_h2o_0_30cm\n"
        "test_001,16.866,96.195,0.65,14.2,6.5\n"
        "test_002,17.201,95.739,0.45,10.1,5.8\n"
    ).encode("utf-8")

    res = client.post(
        "/api/v1/pipeline/run",
        files={"file": ("test.csv", csv_content, "text/csv")}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["total_rows"] == 2
    assert len(body["rows"]) == 2

    row1 = body["rows"][0]
    assert row1["meta"]["index"] == "test_001"
    assert row1["meta"]["lat"] == 16.866
    assert row1["meta"]["lon"] == 96.195
    assert len(row1["predictions"]) == 40
    assert "crop_health_score" in row1["predictions"]
    assert "crop_recommender" in row1["composite_features"]
    assert "crop_health" in row1["composite_features"]
    assert "economic_roi" in row1["composite_features"]
    assert "risk_alerts" in row1["composite_features"]
    assert "land_use" in row1["composite_features"]


def test_pipeline_invalid_file_extension(client: TestClient) -> None:
    res = client.post(
        "/api/v1/pipeline/run",
        files={"file": ("test.png", b"fake_png_data", "image/png")}
    )
    assert res.status_code == 400
