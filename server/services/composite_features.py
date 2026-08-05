"""Transparent composite views over explicit model outputs.

These functions calculate composite feature summaries (crop recommender, health, risk alerts, land use)
from model prediction outputs and optional raw features.
"""

from __future__ import annotations

from typing import Any
from server.config import CROPS, SUITABILITY_COLORS, SUITABILITY_WEIGHTS

COMPOSITE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "crop_recommender": tuple(f"crop_suitability_{crop}" for crop in CROPS),
    "crop_health": ("crop_health_score",),
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
        for target in COMPOSITE_DEPENDENCIES.get(composite, ()):
            if target not in seen:
                ordered.append(target)
                seen.add(target)
    return ordered


def _get_val(predictions: dict[str, Any], target: str, default: Any = None) -> Any:
    pred = predictions.get(target)
    if pred is None:
        return default
    if isinstance(pred, dict) and "value" in pred:
        return pred["value"]
    return pred


class CompositeFeaturesEngine:
    @staticmethod
    def build_crop_recommender(
        predictions: dict[str, Any], raw_features: dict[str, Any] | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        raw_features = raw_features or {}
        crop_scores = []

        for crop in CROPS:
            pred_val = _get_val(predictions, f"crop_suitability_{crop}")
            if pred_val is None:
                continue

            lbl = str(pred_val).lower()
            weight = SUITABILITY_WEIGHTS.get(lbl, 0.40)
            suitability_pct = round(weight * 100.0, 1)

            area_pct = float(raw_features.get(f"crop_area_pct_{crop}", 0.0) or 0.0)
            composite_score = suitability_pct + (area_pct * 0.1)

            crop_scores.append({
                "crop": crop,
                "suitability": lbl,
                "suitability_score": suitability_pct,
                "composite_score": composite_score,
                "color_code": SUITABILITY_COLORS.get(lbl, "#3B82F6")
            })

        crop_scores.sort(key=lambda x: (x["suitability_score"], x["composite_score"]), reverse=True)

        for idx, item in enumerate(crop_scores):
            base_score = item["suitability_score"]
            area_pct = float(raw_features.get(f"crop_area_pct_{item['crop']}", 0.0) or 0.0)
            adjusted_score = base_score + (area_pct * 0.05) - (idx * 0.1)
            item["suitability_score"] = round(max(1.0, min(100.0, adjusted_score)), 1)

        return crop_scores[:top_k]

    @staticmethod
    def build_crop_health_layer(
        predictions: dict[str, Any], raw_features: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        raw_features = raw_features or {}
        score_val = _get_val(predictions, "crop_health_score", 0.6)
        try:
            score = float(score_val)
        except (ValueError, TypeError):
            score = 0.6

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

        ndvi = raw_features.get("ndvi_median_mean") or raw_features.get("ndvi_median")
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
        predictions: dict[str, Any]
    ) -> dict[str, Any]:
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
        predictions: dict[str, Any]
    ) -> dict[str, Any]:
        flood = str(_get_val(predictions, "flood_risk_level", "low")).lower()
        heat = str(_get_val(predictions, "heat_stress_risk", "low")).lower()
        erosion = str(_get_val(predictions, "soil_erosion_risk", "low")).lower()
        
        try:
            drought = float(_get_val(predictions, "drought_risk_score", 0.0))
        except (ValueError, TypeError):
            drought = 0.0

        try:
            water = float(_get_val(predictions, "water_scarcity_risk", 0.0))
        except (ValueError, TypeError):
            water = 0.0

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
            return scores.get(value, 0.0)

        scores = {
            "flood": class_score(flood),
            "drought": drought,
            "heat": class_score(heat),
            "erosion": class_score(erosion),
            "water_scarcity": water,
        }
        maximum = max(scores.values()) if scores else 0.0
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
        predictions: dict[str, Any], raw_features: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        raw_features = raw_features or {}
        try:
            conversion = float(_get_val(predictions, "agricultural_land_conversion_risk", 0.0))
        except (ValueError, TypeError):
            conversion = 0.0

        try:
            urban = float(_get_val(predictions, "urban_encroachment_risk", 0.0))
        except (ValueError, TypeError):
            urban = 0.0

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
        names: list[str] | None,
        predictions: dict[str, Any],
        raw_features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not names:
            names = ["crop_recommender", "crop_health", "economic_roi", "risk_alerts", "land_use"]
        
        result: dict[str, Any] = {}
        for name in names:
            if name == "crop_recommender":
                result[name] = cls.build_crop_recommender(predictions, raw_features)
            elif name == "crop_health":
                result[name] = cls.build_crop_health_layer(predictions, raw_features)
            elif name == "economic_roi":
                result[name] = cls.build_economic_roi_calculator(predictions)
            elif name == "risk_alerts":
                result[name] = cls.build_multi_hazard_risk_alert(predictions)
            elif name == "land_use":
                result[name] = cls.build_land_use_pattern(predictions, raw_features)
        return result
