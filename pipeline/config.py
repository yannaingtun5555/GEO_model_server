#!/usr/bin/env python3
"""
config.py — Configuration and metadata definitions for the 40-model ML Inference Pipeline
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
GP_MODELS_DIR = PROJECT_ROOT / "gp_models"
DATA_FILE    = PROJECT_ROOT / "data" / "combined" / "combined_dataset.csv"

# 17 Crops evaluated for suitability
CROPS = [
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
]

# Suitability rank weights for scoring
SUITABILITY_WEIGHTS = {
    "excellent": 1.0,
    "good": 0.75,
    "moderate": 0.40,
    "poor": 0.10,
}

# Suitability status color codes for Web Map Layers
SUITABILITY_COLORS = {
    "excellent": "#10B981",  # Vibrant Green
    "good": "#3B82F6",       # Blue
    "moderate": "#F59E0B",   # Amber / Orange
    "poor": "#EF4444",       # Red
}

# Health status color codes
HEALTH_COLORS = {
    "Excellent": "#10B981",
    "Good": "#3B82F6",
    "Moderate": "#F59E0B",
    "Poor": "#EF4444",
    "Critical": "#8B5CF6",
}

# 40 Targets defined in prediction_req.md
TARGET_DEFINITIONS = {
    # 1-17: Crop Suitabilities
    **{f"crop_suitability_{crop}": {"type": "classification", "group": "suitability", "crop": crop} for crop in CROPS},

    # 18-20: Core Agronomic Indicators
    "crop_health_score": {"type": "regression", "group": "health"},
    "crop_yield_t_ha": {"type": "regression", "group": "yield"},
    "irrigation_need": {"type": "classification", "group": "irrigation"},

    # 21-23: Climate Forecasts
    "current_month_precipitation": {"type": "regression", "group": "climate"},
    "current_month_mean_temperature": {"type": "regression", "group": "climate"},
    "current_month_mean_solar_rad": {"type": "regression", "group": "climate"},

    # 24-26: Environmental Hazards
    "flood_risk_level": {"type": "classification", "group": "hazard"},
    "drought_risk_score": {"type": "regression", "group": "hazard"},
    "heat_stress_risk": {"type": "classification", "group": "hazard"},

    # 27-30: Management & Soil Risks
    "optimal_planting_month": {"type": "classification", "group": "management"},
    "nitrogen_requirement_level": {"type": "classification", "group": "management"},
    "phosphorus_requirement_level": {"type": "classification", "group": "management"},
    "soil_erosion_risk": {"type": "classification", "group": "hazard"},

    # 31-34: Market & Infrastructure
    "market_integration_score": {"type": "regression", "group": "market"},
    "post_harvest_loss_risk": {"type": "classification", "group": "market"},
    "supply_chain_efficiency": {"type": "regression", "group": "market"},
    "cold_chain_potential": {"type": "regression", "group": "market"},

    # 35-36: Urban & Land Use Change
    "agricultural_land_conversion_risk": {"type": "classification", "group": "land_use"},
    "urban_encroachment_risk": {"type": "classification", "group": "land_use"},

    # 37-39: Water Access & Scarcity
    "irrigation_potential": {"type": "regression", "group": "water"},
    "surface_water_occurrence": {"type": "regression", "group": "water"},
    "water_scarcity_risk": {"type": "classification", "group": "water"},

    # 40: Economic Output
    "agricultural_gdp_forecast": {"type": "regression", "group": "economic"},
}
