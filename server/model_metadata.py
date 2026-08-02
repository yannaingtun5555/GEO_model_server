"""Canonical, truthful metadata for the 40 locally trained surrogate models."""

from __future__ import annotations

from server.config import CROPS, MODEL_TARGETS


SUITABILITY_CLASSES = ["poor", "moderate", "good", "excellent"]

TARGET_METADATA: dict[str, dict] = {
    **{
        f"crop_suitability_{crop}": {
            "display_name": f"{crop.replace('_', ' ').title()} suitability",
            "task_type": "classification",
            "unit": "suitability_class",
            "expected_classes": SUITABILITY_CLASSES,
            "group": "crop_suitability",
        }
        for crop in CROPS
    },
    "crop_health_score": {
        "display_name": "Crop health surrogate score",
        "task_type": "regression",
        "unit": "score_0_to_1",
        "value_range": [0.0, 1.0],
        "group": "crop_health",
    },
    "crop_yield_t_ha": {
        "display_name": "Crop yield surrogate estimate",
        "task_type": "regression",
        "unit": "tonnes_per_hectare",
        "value_range": [0.0, None],
        "group": "yield",
    },
    "irrigation_need": {
        "display_name": "Irrigation need class",
        "task_type": "classification",
        "unit": "class_0_to_2",
        "expected_classes": [0, 1, 2],
        "group": "farm_management",
    },
    "current_month_precipitation_mm": {
        "display_name": "Current-month precipitation reconstruction",
        "task_type": "regression",
        "unit": "millimetres",
        "value_range": [0.0, None],
        "group": "climate",
    },
    "current_month_mean_temperature_c": {
        "display_name": "Current-month mean temperature reconstruction",
        "task_type": "regression",
        "unit": "degrees_celsius",
        "value_range": [-80.0, 80.0],
        "group": "climate",
    },
    "current_month_solar_rad_mj_m2_day": {
        "display_name": "Current-month solar radiation reconstruction",
        "task_type": "regression",
        "unit": "megajoules_per_square_metre_per_day",
        "value_range": [0.0, None],
        "group": "climate",
    },
    "flood_risk_level": {
        "display_name": "Flood risk surrogate class",
        "task_type": "classification",
        "unit": "class_0_to_2",
        "expected_classes": [0, 1, 2],
        "group": "risk",
    },
    "drought_risk_score": {
        "display_name": "Drought risk surrogate score",
        "task_type": "regression",
        "unit": "score_0_to_1",
        "value_range": [0.0, 1.0],
        "group": "risk",
    },
    "heat_stress_risk": {
        "display_name": "Heat-stress risk surrogate class",
        "task_type": "classification",
        "unit": "class_0_to_2",
        "expected_classes": [0, 1, 2],
        "group": "risk",
    },
    "optimal_planting_month": {
        "display_name": "Optimal planting month surrogate class",
        "task_type": "classification",
        "unit": "month_1_to_12",
        "expected_classes": list(range(1, 13)),
        "group": "farm_management",
    },
    "nitrogen_requirement_level": {
        "display_name": "Nitrogen requirement surrogate class",
        "task_type": "classification",
        "unit": "class_0_to_2",
        "expected_classes": [0, 1, 2],
        "group": "farm_management",
    },
    "phosphorus_requirement_level": {
        "display_name": "Phosphorus requirement surrogate class",
        "task_type": "classification",
        "unit": "class_0_to_2",
        "expected_classes": [0, 1, 2],
        "group": "farm_management",
    },
    "soil_erosion_risk": {
        "display_name": "Soil erosion risk surrogate class",
        "task_type": "classification",
        "unit": "class_0_to_2",
        "expected_classes": [0, 1, 2],
        "group": "risk",
    },
}

for target in (
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
):
    TARGET_METADATA[target] = {
        "display_name": target.replace("_", " ").title() + " surrogate score",
        "task_type": "regression",
        "unit": "score_0_to_1",
        "value_range": [0.0, 1.0],
        "group": (
            "market"
            if target
            in {
                "market_integration_score",
                "post_harvest_loss_risk",
                "supply_chain_efficiency",
                "cold_chain_potential",
            }
            else "land_use"
            if target
            in {"agricultural_land_conversion_risk", "urban_encroachment_risk"}
            else "water"
            if target
            in {"irrigation_potential", "surface_water_occurrence", "water_scarcity_risk"}
            else "economic"
        ),
    }

if set(TARGET_METADATA) != set(MODEL_TARGETS):
    missing = sorted(set(MODEL_TARGETS) - set(TARGET_METADATA))
    extra = sorted(set(TARGET_METADATA) - set(MODEL_TARGETS))
    raise RuntimeError(f"Target metadata mismatch; missing={missing}, extra={extra}")
