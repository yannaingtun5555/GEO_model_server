"""API request, response, and error contracts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RowMeta(BaseModel):
    index: str
    sample_id: Optional[str] = None
    lat: float
    lon: float
    region: Optional[str] = None


class RowPredictionResult(BaseModel):
    meta: RowMeta
    predictions: Dict[str, Any]
    composite_features: Dict[str, Any]


class PipelineResponse(BaseModel):
    status: str = "success"
    total_rows: int
    rows: List[RowPredictionResult]
    pipeline_metadata: Dict[str, Any]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    retryable: bool
    details: Optional[List[Dict[str, Any]]] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
