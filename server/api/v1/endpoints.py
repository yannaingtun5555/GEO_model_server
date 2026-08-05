"""Clean Model-Serving API v1 endpoints for dataset CSV processing & prediction."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import pandas as pd
import psutil
from fastapi import APIRouter, File, HTTPException, UploadFile, Request

from server.config import (
    API_VERSION,
    CONTRACT_VERSION,
    CROPS,
    MODEL_TARGETS,
)
from server.contracts import PipelineResponse
from server.core.catalog import model_catalog
from server.core.model_loader import ModelUnavailable, model_manager
from server.core.preprocessor import csv_preprocessor
from server.services.composite_features import CompositeFeaturesEngine
from pipeline.estimator_fallback import estimate_fallback

router = APIRouter(prefix="/api/v1", tags=["Model Server API"])


def _predict_single_target(target: str, feature_row: Dict[str, Any]) -> Dict[str, Any]:
    """Predicts a single target using loaded ML model artifact or domain heuristic fallback."""
    try:
        artifact, metadata = model_manager.get_model(target)
        model = artifact["model"]
        feature_names = artifact.get("features", [])
        
        # Build feature DataFrame row
        x_vals = []
        for name in feature_names:
            val = feature_row.get(name, 0.0)
            try:
                val_flt = float(val)
                if pd.isna(val_flt):
                    val_flt = 0.0
            except (ValueError, TypeError):
                val_flt = 0.0
            x_vals.append(val_flt)

        input_frame = pd.DataFrame([x_vals], columns=feature_names)
        raw_pred = model.predict(input_frame)[0]

        label_encoder = artifact.get("label_encoder")
        if label_encoder is not None:
            try:
                decoded = label_encoder.inverse_transform([int(raw_pred)])[0]
                label_str = str(decoded)
            except Exception:
                label_str = str(raw_pred)
            return {
                "value": label_str,
                "task_type": "classification",
                "label": label_str,
                "is_fallback": False,
            }
        else:
            val_flt = float(raw_pred)
            return {
                "value": val_flt,
                "task_type": "regression",
                "label": f"{val_flt:.2f}",
                "is_fallback": False,
            }
    except Exception:
        # Seamless fallback
        fb = estimate_fallback(target, feature_row)
        return {
            "value": fb.get("value"),
            "task_type": "heuristic",
            "label": str(fb.get("label", fb.get("value"))),
            "is_fallback": True,
        }


@router.post("/pipeline/run", response_model=PipelineResponse, summary="Process CSV dataset, run all 40 predictions, and generate composite features per row")
async def run_pipeline(
    file: UploadFile = File(..., description="Daily GEE/dataset CSV export matching data/test/*.csv or data/raw/yangon/yangon.csv")
) -> PipelineResponse:
    """
    Main API endpoint:
    - Accepts daily uploaded CSV file (e.g. data/test/yangon.csv or data/raw/yangon/yangon.csv).
    - Preprocesses and aligns features per row.
    - Executes predictions for all 40 ML models.
    - Calculates all 5 composite feature groups per row.
    - Returns structured prediction response directly for every land index.
    """
    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    start_time = time.perf_counter()

    try:
        contents = await file.read()
        processed_rows = csv_preprocessor.process_csv(contents)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid or corrupted CSV dataset: {exc}")

    output_rows = []
    models_used_count = 0
    fallbacks_used_count = 0

    for item in processed_rows:
        meta = item["meta"]
        features = item["features"]

        predictions = {}
        for target in MODEL_TARGETS:
            pred_res = _predict_single_target(target, features)
            predictions[target] = pred_res
            if pred_res.get("is_fallback"):
                fallbacks_used_count += 1
            else:
                models_used_count += 1

        composites = CompositeFeaturesEngine.build_requested(None, predictions, features)

        output_rows.append({
            "meta": meta,
            "predictions": predictions,
            "composite_features": composites,
        })

    exec_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    return PipelineResponse(
        status="success",
        total_rows=len(output_rows),
        rows=output_rows,
        pipeline_metadata={
            "filename": file.filename,
            "execution_time_ms": exec_latency_ms,
            "total_predictions_evaluated": len(output_rows) * len(MODEL_TARGETS),
            "models_used_count": models_used_count,
            "fallbacks_used_count": fallbacks_used_count,
        },
    )


@router.get("/live", summary="Server Liveness Probe")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/ready", summary="Server Readiness Probe")
async def readiness() -> dict[str, Any]:
    return {
        "status": "ready",
        "catalog_version": getattr(model_catalog, "catalog_version", "v1"),
        "model_targets_count": len(MODEL_TARGETS),
    }


@router.get("/health", summary="Microservice Resource & Health Diagnostics")
async def health_diagnostics() -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    ram_mb = round(process.memory_info().rss / (1024 * 1024), 1)

    return {
        "status": "healthy",
        "service": "Myanmar Agricultural Model Server",
        "version": API_VERSION,
        "ram_usage_mb": ram_mb,
        "loaded_models_in_ram": len(model_manager._lru_cache),
        "available_targets_count": len(MODEL_TARGETS),
    }


@router.get("/models", summary="List Available 40 Prediction Model Targets")
async def list_models() -> dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "contract_version": CONTRACT_VERSION,
        "total_targets": len(MODEL_TARGETS),
        "crops": list(CROPS),
        "targets": list(MODEL_TARGETS),
    }
