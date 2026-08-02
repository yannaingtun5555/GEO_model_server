"""Strict API v1 request, response, and error contracts."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from server.config import COMPOSITE_FEATURES, MAX_TARGETS_PER_REQUEST, MODEL_TARGETS


YEAR_MONTH_PATTERN = re.compile(r"^(20\d{2})-(0[1-9]|1[0-2])$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionRequest(StrictModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    sample_id: str | None = Field(default=None, min_length=1, max_length=160)
    region_name: str | None = None
    system_index: str | None = None
    lat: float | None = Field(default=None, ge=9, le=29)
    lon: float | None = Field(default=None, ge=92, le=102)
    observation_month: str | None = None
    targets: list[str] | None = Field(default=None, min_length=1, max_length=MAX_TARGETS_PER_REQUEST)
    include_all_targets: bool = False
    composite_features: list[str] = Field(default_factory=list, max_length=len(COMPOSITE_FEATURES))

    @model_validator(mode="after")
    def validate_contract(self) -> "PredictionRequest":
        uses_sample = self.sample_id is not None
        coordinate_values = (self.lat, self.lon, self.observation_month)
        uses_coordinates = any(value is not None for value in coordinate_values)
        uses_legacy = self.region_name is not None or self.system_index is not None
        if uses_sample + uses_coordinates + uses_legacy != 1:
            raise ValueError(
                "provide exactly one locator: sample_id, region_name, system_index, or lat+lon"
            )
        if uses_coordinates:
            if self.lat is None or self.lon is None:
                raise ValueError("lat and lon must be provided together")
            if self.observation_month is not None and not YEAR_MONTH_PATTERN.fullmatch(self.observation_month):
                raise ValueError("observation_month must use YYYY-MM format")

        if self.include_all_targets and self.targets:
            raise ValueError("targets and include_all_targets cannot be combined")
        if not self.include_all_targets and not self.targets and not self.composite_features:
            raise ValueError(
                "provide targets, composite_features, or set include_all_targets=true"
            )
        if self.targets:
            unknown_targets = sorted(set(self.targets) - set(MODEL_TARGETS))
            if unknown_targets:
                raise ValueError(f"unknown targets: {unknown_targets}")
            if len(set(self.targets)) != len(self.targets):
                raise ValueError("targets must not contain duplicates")

        unknown_composites = sorted(set(self.composite_features) - set(COMPOSITE_FEATURES))
        if unknown_composites:
            raise ValueError(f"unknown composite_features: {unknown_composites}")
        if len(set(self.composite_features)) != len(self.composite_features):
            raise ValueError("composite_features must not contain duplicates")
        return self


class LocationResponse(StrictModel):
    sample_id: str
    grid_id: str
    region: str
    observation_month: str
    requested_lat: float | None
    requested_lon: float | None
    matched_lat: float
    matched_lon: float
    distance_km: float = Field(ge=0)


class TargetPrediction(StrictModel):
    value: float | int | str
    label: str | None
    unit: str
    task_type: Literal["classification", "regression"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_kind: Literal["random_forest_vote_share_uncalibrated"] | None
    probabilities: dict[str, float] | None
    model_version: str
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_schema_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_source: Literal["primary", "prototype"]
    deployment_status: Literal["experimental"]
    validation_status: Literal["healthy", "flagged", "unknown"]
    warnings: list[str]

    @model_validator(mode="after")
    def validate_confidence_semantics(self) -> "TargetPrediction":
        if self.task_type == "classification":
            if self.confidence_kind != "random_forest_vote_share_uncalibrated":
                raise ValueError("classifications must identify uncalibrated tree-vote confidence")
        elif self.confidence_kind is not None:
            raise ValueError("regressions cannot declare classification confidence semantics")
        return self


class ProvenanceResponse(StrictModel):
    feature_dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    spatial_index_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_source: str | None
    source_date: str | None
    source_version: str | None
    quality_flag: int | None
    label_source: Literal["rule_engineered_surrogate"]
    field_validated: Literal[False]


class ExecutionMetadata(StrictModel):
    response_time_ms: float = Field(ge=0)
    queue_wait_ms: float = Field(ge=0)
    cached: bool
    models_loaded_count: int = Field(ge=0)


class PredictionResponse(StrictModel):
    api_version: Literal["v1"]
    contract_version: Literal["model-inference-v1"]
    catalog_version: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_id: str
    status: Literal["success"]
    location: LocationResponse
    predictions: dict[str, TargetPrediction]
    composite_features: dict[str, Any]
    provenance: ProvenanceResponse
    execution_metadata: ExecutionMetadata


class ErrorDetail(StrictModel):
    code: str
    message: str
    request_id: str
    retryable: bool
    details: list[dict[str, Any]] | None = None


class ErrorResponse(StrictModel):
    error: ErrorDetail
