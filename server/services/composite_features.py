#!/usr/bin/env python3
"""
server/services/composite_features.py — Composite Multi-Model Intelligence Features Engine
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from server.config import CROPS, SUITABILITY_WEIGHTS, SUITABILITY_COLORS

class CompositeFeaturesEngine:
    """
    Combines outputs across all 40 ML models to generate high-level actionable insights:
    1. Micro-Regional Crop Recommender
    2. Geospatial Crop Health Layer (NDVI)
    3. Economic Yield & ROI Calculator
    4. Crop Health Map Layer
    5. Multi-Hazard Risk Alert Engine
    6. Land Use Pattern & Conversion Risk Engine
    """

    @staticmethod
    def build_crop_recommender(predictions: Dict[str, Any], raw_features: Dict[str, Any] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Ranks best crops to plant based on suitability predictions & regional crop historical suitability."""
        raw_features = raw_features or {}
        crop_scores = []
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
            "health_score_pct": health_pct,
            "ndvi_median": ndvi,
            "health_status": status,
            "map_color_hex": color
        }

    @staticmethod
    def build_economic_roi_calculator(predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates estimated crop yield productivity and economic ROI score."""
        yield_t_ha = float(predictions.get("crop_yield_t_ha", 3.20))
        gdp_forecast = float(predictions.get("agricultural_gdp_forecast", 0.65))
        market_integration = float(predictions.get("market_integration_score", 0.70))

        # Composite economic rating
        roi_score = (yield_t_ha / 5.0 * 0.50) + (gdp_forecast * 0.25) + (market_integration * 0.25)
        roi_pct = round(min(1.0, roi_score) * 100.0, 1)

        if roi_pct >= 75.0:
            roi_rating = "HIGH ROI"
        elif roi_pct >= 50.0:
            roi_rating = "MODERATE ROI"
        else:
            roi_rating = "LOW ROI"

        return {
            "projected_yield_t_ha": round(yield_t_ha, 2),
            "market_integration_index": round(market_integration, 2),
            "gdp_growth_index": round(gdp_forecast, 2),
            "economic_roi_score": roi_pct,
            "roi_rating": roi_rating
        }

    @staticmethod
    def build_multi_hazard_risk_alert(predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combines flood, drought, heat stress, soil erosion, and water scarcity risks into alert status."""
        flood = str(predictions.get("flood_risk_level", "0"))
        drought = float(predictions.get("drought_risk_score", 0.15))
        heat = str(predictions.get("heat_stress_risk", "0"))
        erosion = str(predictions.get("soil_erosion_risk", "0"))
        water_scarcity = float(predictions.get("water_scarcity_risk", 0.20))

        flood_score = 0.8 if flood in ["2", "high", "critical"] else (0.4 if flood in ["1", "moderate"] else 0.1)
        heat_score = 0.8 if heat in ["2", "high", "critical"] else (0.4 if heat in ["1", "moderate"] else 0.1)
        erosion_score = 0.8 if erosion in ["2", "high", "critical"] else (0.4 if erosion in ["1", "moderate"] else 0.1)

        max_risk = max(flood_score, drought, heat_score, erosion_score, water_scarcity)

        if max_risk >= 0.70:
            alert_level, color = "CRITICAL HAZARD ALERT", "#EF4444"
        elif max_risk >= 0.40:
            alert_level, color = "WARNING (MODERATE RISK)", "#F59E0B"
        else:
            alert_level, color = "LOW RISK (STABLE)", "#10B981"

        return {
            "overall_alert_level": alert_level,
            "max_risk_score": round(max_risk, 2),
            "flood_risk": "High" if flood_score >= 0.7 else ("Moderate" if flood_score >= 0.3 else "Low"),
            "drought_risk_score": round(drought, 2),
            "heat_stress_risk": "High" if heat_score >= 0.7 else ("Moderate" if heat_score >= 0.3 else "Low"),
            "water_scarcity_risk": round(water_scarcity, 2),
            "alert_color_hex": color
        }

    @staticmethod
    def build_land_use_pattern(predictions: Dict[str, Any], raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates land conversion risk, urban encroachment, and cropland fraction."""
        conversion_risk = float(predictions.get("agricultural_land_conversion_risk", 0.10))
        urban_encroachment = float(predictions.get("urban_encroachment_risk", 0.08))
        cropland_pct = float(raw_features.get("cropland_fraction", 0.85)) * 100.0

        if urban_encroachment >= 0.60 or conversion_risk >= 0.60:
            status = "HIGH URBAN ENCROACHMENT RISK"
        elif urban_encroachment >= 0.30 or conversion_risk >= 0.30:
            status = "MODERATE LAND USE SHIFT"
        else:
            status = "STABLE AGRICULTURAL ZONE"

        return {
            "land_use_status": status,
            "cropland_fraction_pct": round(cropland_pct, 1),
            "conversion_risk_score": round(conversion_risk, 2),
            "urban_encroachment_score": round(urban_encroachment, 2)
        }
