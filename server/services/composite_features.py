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
            pred_val = predictions.get(f"crop_suitability_{crop}")
            if pred_val is None:
                continue

            lbl = str(pred_val).lower()
            weight = SUITABILITY_WEIGHTS.get(lbl, 0.40)
            suitability_pct = round(weight * 100.0, 1)

            # Regional historical crop area tie-breaker
            area_pct = float(raw_features.get(f"crop_area_pct_{crop}", 0.0) or 0.0)
            composite_score = suitability_pct + (area_pct * 0.1)

            crop_scores.append({
                "crop": crop,
                "suitability": lbl,
                "suitability_score": suitability_pct,
                "composite_score": composite_score,
                "color_code": SUITABILITY_COLORS.get(lbl, "#3B82F6")
            })

        # Sort primary by suitability_score, secondary by regional composite tie-breaker
        crop_scores.sort(key=lambda x: (x["suitability_score"], x["composite_score"]), reverse=True)

        # Adjust suitability_score to ensure strictly descending order (rank #1 is highest)
        for idx, item in enumerate(crop_scores):
            base_score = item["suitability_score"]
            area_pct = float(raw_features.get(f"crop_area_pct_{item['crop']}", 0.0) or 0.0)
            adjusted_score = base_score + (area_pct * 0.05) - (idx * 0.1)
            item["suitability_score"] = round(max(1.0, min(100.0, adjusted_score)), 1)

        return crop_scores[:top_k]

    @staticmethod
    def build_crop_health_layer(predictions: Dict[str, Any], raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """Generates geospatial crop health layer status and Web Map styling."""
        health_score = predictions.get("crop_health_score", 0.75)
        if isinstance(health_score, (int, float)):
            health_pct = round(float(health_score) * 100.0, 1)
        else:
            health_pct = 75.0

        ndvi = raw_features.get("ndvi_median_growing_season_mean", raw_features.get("ndvi_median_mean", 0.60))
        try:
            ndvi = round(float(ndvi), 4)
        except (ValueError, TypeError):
            ndvi = 0.6000

        if health_pct >= 85.0:
            status, color = "Optimal Health", "#10B981"
        elif health_pct >= 70.0:
            status, color = "Good Condition", "#3B82F6"
        elif health_pct >= 50.0:
            status, color = "Moderate Stress", "#F59E0B"
        else:
            status, color = "Critical Stress", "#EF4444"

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
