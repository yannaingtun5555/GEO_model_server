"""Validated runtime configuration for the model-serving process."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_local_env(path: Path) -> None:
    """Load simple KEY=VALUE entries without executing shell content."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or not key[0].isalpha():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_local_env(PROJECT_ROOT / ".env")


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value")


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}")
    return value


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}")
    return value


ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
if ENVIRONMENT not in {"development", "test", "production"}:
    raise RuntimeError("ENVIRONMENT must be development, test, or production")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = _int_env("PORT", 8001)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
SERVICE_NAME = "myanmar-agricultural-model-server"
SERVICE_VERSION = "2.0.0"
API_VERSION = "v1"
CONTRACT_VERSION = "model-inference-v1"

MODELS_DIR = _path_env("MODELS_DIR", PROJECT_ROOT / "models")
PROTOTYPES_DIR = _path_env("PROTOTYPES_DIR", PROJECT_ROOT / "models_prototypes")
MODEL_MANIFEST_FILE = _path_env("MODEL_MANIFEST_FILE", MODELS_DIR / "manifest.json")
FEATURE_DATA_FILE = _path_env(
    "FEATURE_DATA_FILE", PROJECT_ROOT / "data" / "processed" / "features_serving.parquet"
)
SPATIAL_INDEX_FILE = _path_env(
    "SPATIAL_INDEX_FILE", PROJECT_ROOT / "data" / "processed" / "spatial_index.parquet"
)

AUTH_REQUIRED = _bool_env("AUTH_REQUIRED", ENVIRONMENT == "production")
MODEL_SERVER_API_KEY = os.getenv("MODEL_SERVER_API_KEY", "").strip()
if AUTH_REQUIRED and len(MODEL_SERVER_API_KEY) < 24:
    raise RuntimeError(
        "MODEL_SERVER_API_KEY must contain at least 24 characters when AUTH_REQUIRED=true"
    )

CORS_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
)
if ENVIRONMENT == "production" and "*" in CORS_ORIGINS:
    raise RuntimeError("Wildcard CORS is forbidden in production")

MAX_LOADED_MODELS = _int_env("MAX_LOADED_MODELS", 2)
MAX_RAM_MB = _int_env("MAX_RAM_MB", 2048)
MODEL_MEMORY_EXPANSION_FACTOR = _float_env(
    "MODEL_MEMORY_EXPANSION_FACTOR", 2.25, 1.0
)
# One inference at a time per process keeps references to evicted large forests
# from overlapping. Scale horizontally with multiple isolated containers.
MAX_CONCURRENT_REQUESTS = _int_env("MAX_CONCURRENT_REQUESTS", 1)
QUEUE_TIMEOUT_SECONDS = _float_env("QUEUE_TIMEOUT_SECONDS", 5.0, 0.1)
REQUEST_EXECUTION_TIMEOUT_SECONDS = _float_env(
    "REQUEST_EXECUTION_TIMEOUT_SECONDS", 30.0, 1.0
)
MAX_NEAREST_DISTANCE_KM = _float_env("MAX_NEAREST_DISTANCE_KM", 8.0, 0.1)
MAX_TARGETS_PER_REQUEST = _int_env("MAX_TARGETS_PER_REQUEST", 40)
MAX_EXPANDED_SYNC_TARGETS = _int_env(
    "MAX_EXPANDED_SYNC_TARGETS", 17 if ENVIRONMENT == "production" else 40
)

ALLOW_PROTOTYPE_MODELS = _bool_env("ALLOW_PROTOTYPE_MODELS", False)
ALLOW_EXPERIMENTAL_RELEASE = _bool_env("ALLOW_EXPERIMENTAL_RELEASE", False)
VERIFY_MODEL_CHECKSUMS_ON_STARTUP = _bool_env(
    "VERIFY_MODEL_CHECKSUMS_ON_STARTUP", ENVIRONMENT == "production"
)
if ENVIRONMENT == "production":
    if not AUTH_REQUIRED:
        raise RuntimeError("AUTH_REQUIRED must be true in production")
    if not VERIFY_MODEL_CHECKSUMS_ON_STARTUP:
        raise RuntimeError("VERIFY_MODEL_CHECKSUMS_ON_STARTUP must be true in production")
    if ALLOW_PROTOTYPE_MODELS:
        raise RuntimeError("ALLOW_PROTOTYPE_MODELS must be false in production")
    if MAX_CONCURRENT_REQUESTS != 1:
        raise RuntimeError(
            "MAX_CONCURRENT_REQUESTS must be 1 per process; scale with isolated replicas"
        )

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = _int_env("REDIS_PORT", 6380)
REDIS_DB = _int_env("REDIS_DB", 0, 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
CACHE_TTL_SECONDS = _int_env("CACHE_TTL_SECONDS", 86400)
MAX_CACHE_ENTRIES = _int_env("MAX_CACHE_ENTRIES", 1000)

REGIONS = ("ayeyawaddy", "bago", "magway", "mandalay", "sagaing", "yangon")
CROPS = (
    "monsoon_rice",
    "dry_season_rice",
    "maize",
    "sugarcane",
    "cassava",
    "durian",
    "mangosteen",
    "longan",
    "mango",
    "chili",
    "tomato",
    "black_gram",
    "green_gram",
    "pigeon_pea",
    "groundnut",
    "sesame",
    "rubber",
)

MODEL_TARGETS = (
    *(f"crop_suitability_{crop}" for crop in CROPS),
    "crop_health_score",
    "crop_yield_t_ha",
    "irrigation_need",
    "current_month_precipitation_mm",
    "current_month_mean_temperature_c",
    "current_month_solar_rad_mj_m2_day",
    "flood_risk_level",
    "drought_risk_score",
    "heat_stress_risk",
    "optimal_planting_month",
    "nitrogen_requirement_level",
    "phosphorus_requirement_level",
    "soil_erosion_risk",
    "market_integration_score",
    "post_harvest_loss_risk",
    "supply_chain_efficiency",
    "cold_chain_potential",
    "agricultural_land_conversion_risk",
    "urban_encroachment_risk",
    "irrigation_potential",
    "surface_water_occurrence",
    "water_scarcity_risk",
    "agricultural_gdp_forecast",
)

COMPOSITE_FEATURES = (
    "crop_recommender",
    "crop_health",
    "economic_roi",
    "risk_alerts",
    "land_use",
)

SUITABILITY_WEIGHTS = {
    "excellent": 1.0,
    "good": 0.75,
    "moderate": 0.40,
    "poor": 0.10,
}

SUITABILITY_COLORS = {
    "excellent": "#10B981",
    "good": "#3B82F6",
    "moderate": "#F59E0B",
    "poor": "#EF4444",
}
