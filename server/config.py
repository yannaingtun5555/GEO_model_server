#!/usr/bin/env python3
"""
server/config.py — Configuration for Model Serving Microservice
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
MODELS_DIR        = PROJECT_ROOT / "models"
PROTOTYPES_DIR    = PROJECT_ROOT / "models_prototypes"
DATA_FILE         = PROJECT_ROOT / "data" / "combined" / "combined_dataset.csv"
STORAGE_DIR       = PROJECT_ROOT / "data" / "regional_storage"

# Memory & LRU Model Loading Limits
MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", "4"))    # Keep max 4 heavy models in RAM (in normal mode)
MAX_RAM_MB        = int(os.getenv("MAX_RAM_MB", "2048"))        # 2 GB RAM limit cap

# Boost Mode & High-Performance Concurrency Settings
BOOST_MODE              = os.getenv("BOOST_MODE", "false").lower() in ("true", "1", "t") # Preloads & keeps all 40 models in RAM
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))                # Max active prediction workers
QUEUE_TIMEOUT_SECONDS   = float(os.getenv("QUEUE_TIMEOUT_SECONDS", "30.0"))             # Max request queue wait time

# Redis Cache Settings
REDIS_HOST        = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT        = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB          = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD    = os.getenv("REDIS_PASSWORD", None)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24 Hours TTL

# Regions
REGIONS = ["ayeyawaddy", "bago", "magway", "mandalay", "sagaing", "yangon"]

# 17 Crops
CROPS = [
    "monsoon_rice", "dry_season_rice", "maize", "sugarcane", "cassava",
    "durian", "mangosteen", "longan", "mango", "chili", "tomato",
    "black_gram", "green_gram", "pigeon_pea", "groundnut", "sesame", "rubber"
]

SUITABILITY_WEIGHTS = {
    "excellent": 1.0,
    "good": 0.75,
    "moderate": 0.40,
    "poor": 0.10,
}

SUITABILITY_COLORS = {
    "excellent": "#10B981",  # Green
    "good": "#3B82F6",       # Blue
    "moderate": "#F59E0B",   # Amber
    "poor": "#EF4444",       # Red
}
