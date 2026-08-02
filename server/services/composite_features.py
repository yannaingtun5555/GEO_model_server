"""Transparent composite views over explicit model outputs.

These functions never substitute guessed values.  A composite is either computed from
all declared dependencies or returned as unavailable with a machine-readable reason.
"""

from __future__ import annotations

from typing import Any

from server.config import CROPS, SUITABILITY_COLORS, SUITABILITY_WEIGHTS


COMPOSITE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "crop_recommender": tuple(f"crop_suitability_{crop}" for crop in CROPS),
    "crop_health": ("crop_health_score",),
    # ROI is explicitly unavailable until verified prices/costs exist. Do not
    # spend RAM/CPU on unrelated surrogate inference merely to report that fact.
    "economic_roi": (),
    "risk_alerts": (
        "flood_risk_level",
        "drought_risk_score",
        "heat_stress_risk",
        "soil_erosion_risk",
        "water_scarcity_risk",
    ),
    "land_use": (
        "agricultural_land_conversion_risk",
        "urban_encroachment_risk",
    ),
}


def resolve_targets(requested: list[str], composites: list[str]) -> list[str]:
    ordered = list(requested)
    seen = set(ordered)
    for composite in composites:
        for target in COMPOSITE_DEPENDENCIES[composite]:
            if target not in seen:
                ordered.append(target)
                seen.add(target)
    return ordered


def _value(predictions: dict[str, dict[str, Any]], target: str) -> Any:
    return predictions[target]["value"]


class CompositeFeaturesEngine:
    @staticmethod
    def build_crop_recommender(
        predictions: dict[str, dict[str, Any]], top_k: int = 5
    ) -> dict[str, Any]:
        tier_order = ("excellent", "good", "moderate", "poor")
        tiers: dict[str, list[dict[str, Any]]] = {label: [] for label in tier_order}
        for crop in CROPS:
            target = f"crop_suitability_{crop}"
            label = str(_value(predictions, target)).lower()
            if label not in SUITABILITY_WEIGHTS:
                raise ValueError(f"{target} returned unsupported suitability label '{label}'")
            tiers[label].append(
                {
                    "crop": crop,
                    "suitability": label,
                    "tree_vote_agreement": predictions[target].get("confidence"),
                    "color_code": SUITABILITY_COLORS[label],
                }
            )
        for crops in tiers.values():
            crops.sort(key=lambda item: item["crop"])
        top_tier = next((label for label in tier_order if tiers[label]), None)
        top_group = tiers[top_tier][:top_k] if top_tier is not None else []
        return {
            "status": "experimental",
            "strict_ranking_available": False,
            "reason_code": "CROSS_MODEL_CALIBRATION_REQUIRED",
            "recommendation_basis": (
                "shared suitability tier only; crops inside a tier are intentionally tied"
            ),
            "top_suitability_tier": top_tier,
            "top_recommendations": top_group,
            "suitability_tiers": tiers,
            "probability_calibrated": False,
            "field_validated": False,
        }

    @staticmethod
    def build_crop_health_layer(
        predictions: dict[str, dict[str, Any]], raw_features: dict[str, Any]
    ) -> dict[str, Any]:
        score = float(_value(predictions, "crop_health_score"))
        if score >= 0.80:
            status, color = "Excellent", "#10B981"
        elif score >= 0.60:
            status, color = "Good", "#3B82F6"
        elif score >= 0.40:
            status, color = "Moderate", "#F59E0B"
        elif score >= 0.20:
            status, color = "Poor", "#EF4444"
        else:
            status, color = "Critical", "#8B5CF6"
        ndvi = raw_features.get("ndvi_median_mean")
        return {
            "status": "experimental",
            "health_score": score,
            "health_class": status,
            "ndvi_median": float(ndvi) if ndvi is not None else None,
            "map_color_hex": color,
            "field_validated": False,
        }

    @staticmethod
    def build_economic_roi_calculator(
        predictions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        # The current GDP artifact is a 0–1 engineered index and there are no verified
        # farm-gate price/cost inputs.  Calling this ROI would create a false currency value.
        return {
            "status": "unavailable",
            "reason_code": "VERIFIED_ECONOMIC_INPUTS_REQUIRED",
            "message": (
                "ROI is withheld until verified crop price, input cost and currency-period "
                "data are supplied."
            ),
        }

    @staticmethod
    def build_multi_hazard_risk_alert(
        predictions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        flood = str(_value(predictions, "flood_risk_level")).lower()
        heat = str(_value(predictions, "heat_stress_risk")).lower()
        erosion = str(_value(predictions, "soil_erosion_risk")).lower()
        drought = float(_value(predictions, "drought_risk_score"))
        water = float(_value(predictions, "water_scarcity_risk"))

        def class_score(value: str) -> float:
            scores = {
                "0": 0.0,
                "low": 0.0,
                "1": 0.5,
                "medium": 0.5,
                "moderate": 0.5,
                "2": 1.0,
                "high": 1.0,
                "critical": 1.0,
            }
            if value not in scores:
                raise ValueError(f"unsupported risk class '{value}'")
            return scores[value]

        scores = {
            "flood": class_score(flood),
            "drought": drought,
            "heat": class_score(heat),
            "erosion": class_score(erosion),
            "water_scarcity": water,
        }
        maximum = max(scores.values())
        level = "high" if maximum >= 0.7 else "medium" if maximum >= 0.4 else "low"
        return {
            "status": "experimental",
            "overall_level": level,
            "risk_scores": scores,
            "advisory_status": "human_review_required",
            "approved_action": None,
            "field_validated": False,
        }

    @staticmethod
    def build_land_use_pattern(
        predictions: dict[str, dict[str, Any]], raw_features: dict[str, Any]
    ) -> dict[str, Any]:
        conversion = float(_value(predictions, "agricultural_land_conversion_risk"))
        urban = float(_value(predictions, "urban_encroachment_risk"))
        cropland = raw_features.get("cropland_fraction")
        maximum = max(conversion, urban)
        level = "high" if maximum >= 0.6 else "medium" if maximum >= 0.3 else "low"
        return {
            "status": "experimental",
            "risk_level": level,
            "conversion_risk_score": conversion,
            "urban_encroachment_score": urban,
            "cropland_fraction": float(cropland) if cropland is not None else None,
            "field_validated": False,
        }

    @classmethod
    def build_requested(
        cls,
        names: list[str],
        predictions: dict[str, dict[str, Any]],
        raw_features: dict[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in names:
            if name == "crop_recommender":
                result[name] = cls.build_crop_recommender(predictions)
            elif name == "crop_health":
                result[name] = cls.build_crop_health_layer(predictions, raw_features)
            elif name == "economic_roi":
                result[name] = cls.build_economic_roi_calculator(predictions)
            elif name == "risk_alerts":
                result[name] = cls.build_multi_hazard_risk_alert(predictions)
            elif name == "land_use":
                result[name] = cls.build_land_use_pattern(predictions, raw_features)
        return result
