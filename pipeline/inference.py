#!/usr/bin/env python3
"""
inference.py — End-to-End ML Pipeline for Web Backend Integration
===================================================================
Loads all available models from `gp_models/` and `models/`, runs predictions
for all 40 targets (using fallback estimators for missing models), and formats
the outputs into a complete, backend-ready JSON payload.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd
import joblib

from .config import (
    CROPS, TARGET_DEFINITIONS, SUITABILITY_WEIGHTS,
    SUITABILITY_COLORS, HEALTH_COLORS, MODELS_DIR, GP_MODELS_DIR
)
from .estimator_fallback import estimate_fallback


class ModelPipeline:
    """
    40-Model ML Pipeline for Crop Analysis & Agricultural Intelligence.
    Handles dynamic model discovery, inference, fallback estimation, and BE JSON formatting.
    """

    def __init__(self, models_dirs: Optional[List[Path]] = None):
        if models_dirs is None:
            models_dirs = [GP_MODELS_DIR, MODELS_DIR]

        self.models_dirs = [Path(d) for d in models_dirs]
        self.models: Dict[str, Dict[str, Any]] = {}
        self.loaded_targets: List[str] = []
        self.fallback_targets: List[str] = []
        self._load_all_models()

    def _load_all_models(self):
        """Discover and load all .pkl model artifacts from specified model directories."""
        self.models.clear()
        found_files = {}

        for m_dir in self.models_dirs:
            if m_dir.exists():
                for pkl in m_dir.glob("*.pkl"):
                    # Avoid overwriting if already loaded from primary directory
                    target_key = self._extract_target_name(pkl.stem)
                    if target_key not in found_files:
                        found_files[target_key] = pkl

        for target_key, pkl_path in found_files.items():
            try:
                artifact = joblib.load(pkl_path)
                if isinstance(artifact, dict) and "model" in artifact:
                    self.models[target_key] = artifact
                    self.loaded_targets.append(target_key)
            except Exception as e:
                print(f"[WARN] Failed to load model artifact '{pkl_path.name}': {e}")

        # Determine fallbacks
        all_targets = list(TARGET_DEFINITIONS.keys())
        self.fallback_targets = [t for t in all_targets if t not in self.models]

    def _extract_target_name(self, stem: str) -> str:
        """Extract canonical target name from file stem (e.g. crop_suitability_chili_rf_classifier -> crop_suitability_chili)."""
        for suffix in ["_rf_classifier", "_gb_classifier", "_rf_regressor", "_gb_regressor"]:
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
        return stem

    def predict_target(self, target: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict a single target using ML model if loaded, else fallback estimator."""
        if target in self.models:
            artifact = self.models[target]
            model = artifact.get("model")
            feat_list = artifact.get("features", [])
            le = artifact.get("label_encoder")

            # Prepare feature vector
            x_vals = []
            for f in feat_list:
                x_vals.append(features.get(f, 0.0))

            X_in = pd.DataFrame([x_vals], columns=feat_list)
            
            try:
                raw_pred = model.predict(X_in)[0]

                # If classification with label encoder
                if le is not None:
                    if isinstance(raw_pred, (int, np.integer)):
                        label_str = str(le.classes_[raw_pred])
                    else:
                        label_str = str(raw_pred)
                    
                    # Probabilities if available
                    probs = None
                    if hasattr(model, "predict_proba"):
                        p_arr = model.predict_proba(X_in)[0]
                        probs = {str(cls): float(p) for cls, p in zip(le.classes_, p_arr)}

                    return {
                        "value": label_str,
                        "label": label_str.capitalize(),
                        "probabilities": probs,
                        "is_fallback": False
                    }
                else:
                    return {
                        "value": float(raw_pred),
                        "label": f"{raw_pred:.2f}",
                        "is_fallback": False
                    }
            except Exception as e:
                # If model prediction fails at runtime, seamlessly use fallback
                fb = estimate_fallback(target, features)
                fb["fallback_reason"] = f"Model execution error: {e}"
                return fb

        # Fallback estimator
        return estimate_fallback(target, features)

    def predict_all(self, features: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run predictions across all 40 defined targets."""
        predictions = {}
        for target in TARGET_DEFINITIONS.keys():
            predictions[target] = self.predict_target(target, features)
        return predictions

    def process_pipeline(
        self,
        features: Union[Dict[str, Any], pd.Series],
        location_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main pipeline entry point.
        Takes input feature vector and location metadata, performs 40 predictions,
        and generates a complete Web Backend (BE) structured JSON payload.
        """
        start_time = time.time()

        if isinstance(features, pd.Series):
            features_dict = features.to_dict()
        else:
            features_dict = dict(features)

        if location_meta is None:
            location_meta = {
                "latitude": float(features_dict.get("latitude", 16.80)),
                "longitude": float(features_dict.get("longitude", 96.15)),
                "region": str(features_dict.get("region", "Ayeyawaddy")).capitalize(),
                "district": str(features_dict.get("district", "Unknown")),
            }

        # 1. Run 40 Target Predictions
        preds = self.predict_all(features_dict)

        # 2. Extract & Format Modules
        crop_suitabilities = self._format_crop_suitabilities(preds)
        crop_health_layer = self._format_crop_health_layer(preds, features_dict)
        crop_recommendations = self._format_crop_recommendations(crop_suitabilities, preds, features_dict)
        location_detail_panel = self._format_location_detail_panel(features_dict, preds)
        risk_alerts = self._format_risk_alerts(preds, features_dict)
        user_pattern_analysis = self._format_user_pattern_analysis(preds, features_dict)
        market_infrastructure = self._format_market_infrastructure(preds, features_dict)
        gdp_forecast = self._format_gdp_forecast(preds)

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # 3. Assemble BE-Ready Response JSON
        payload = {
            "status": "success",
            "location_metadata": location_meta,
            "crop_health_map_layer": crop_health_layer,
            "crop_recommendation": crop_recommendations,
            "location_detail_panel": location_detail_panel,
            "risk_alerts": risk_alerts,
            "user_pattern_analysis": user_pattern_analysis,
            "market_and_infrastructure": market_infrastructure,
            "agricultural_gdp_forecast": gdp_forecast,
            "crop_suitabilities": crop_suitabilities,
            "raw_predictions": preds,
            "pipeline_metadata": {
                "total_targets_evaluated": len(preds),
                "models_loaded_count": len(self.loaded_targets),
                "fallbacks_used_count": len(self.fallback_targets),
                "execution_latency_ms": latency_ms,
                "models_dir": [str(d) for d in self.models_dirs if d.exists()],
            }
        }
        return payload

    def _format_crop_suitabilities(self, preds: Dict[str, Any]) -> Dict[str, Any]:
        """Format 17 crop suitability predictions with scores, labels, and map hex colors."""
        suitabilities = {}
        for crop in CROPS:
            target_key = f"crop_suitability_{crop}"
            pred_data = preds.get(target_key, {})
            val = str(pred_data.get("value", "moderate")).lower()
            
            # Normalize label
            if val not in SUITABILITY_WEIGHTS:
                val = "moderate"

            score = SUITABILITY_WEIGHTS.get(val, 0.4)
            color = SUITABILITY_COLORS.get(val, "#F59E0B")

            suitabilities[crop] = {
                "crop": crop,
                "suitability_class": val,
                "suitability_label": val.capitalize(),
                "suitability_score": score,
                "color_hex": color,
                "probabilities": pred_data.get("probabilities"),
                "is_fallback": pred_data.get("is_fallback", False)
            }
        return suitabilities

    def _format_crop_health_layer(self, preds: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Crop Health Map Layer feature module."""
        health_pred = preds.get("crop_health_score", {})
        score = float(health_pred.get("value", 0.65))

        if score >= 0.80:
            status = "Excellent"
        elif score >= 0.60:
            status = "Good"
        elif score >= 0.40:
            status = "Moderate"
        elif score >= 0.20:
            status = "Poor"
        else:
            status = "Critical"

        color = HEALTH_COLORS.get(status, "#3B82F6")

        return {
            "health_score": score,
            "health_status": status,
            "status_color_hex": color,
            "ndvi_median": float(features.get("ndvi_median_mean", 0.55)),
            "ndwi_water_index": float(features.get("ndwi_mcf_median_mean", 0.15)),
            "soil_moisture_m3_m3": float(features.get("era5_soil_moisture_m3_m3_mean", 0.28)),
            "map_layer_style": {
                "fill_color": color,
                "opacity": 0.75,
                "stroke_color": "#FFFFFF",
                "stroke_width": 1.5
            }
        }

    def _format_crop_recommendations(
        self,
        suitabilities: Dict[str, Any],
        preds: Dict[str, Any],
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rank crops based on suitability score, predicted yield, and market integration."""
        yield_pred = preds.get("crop_yield_t_ha", {}).get("value", 3.8)
        market_score = preds.get("market_integration_score", {}).get("value", 0.75)

        ranked = []
        for crop, data in suitabilities.items():
            base_score = data["suitability_score"]
            # Combine suitability score with market & yield factors
            final_rank_score = round(base_score * 0.7 + (market_score * 0.3), 3)

            # Crop specific yield estimate
            if "rice" in crop:
                est_yield = round(yield_pred * 1.1, 2)
            elif crop in ["maize", "sugarcane"]:
                est_yield = round(yield_pred * 1.2, 2)
            else:
                est_yield = round(yield_pred * 0.6, 2)

            ranked.append({
                "crop": crop,
                "crop_display_name": crop.replace("_", " ").title(),
                "rank_score": final_rank_score,
                "suitability_class": data["suitability_class"],
                "expected_yield_t_ha": est_yield,
                "suitability_color": data["color_hex"],
                "rationale": f"High agronomic fit for {crop.replace('_', ' ')} with favorable soil & moisture conditions."
            })

        # Sort by rank_score descending
        ranked.sort(key=lambda x: x["rank_score"], reverse=True)

        return {
            "top_recommended_crops": ranked[:5],
            "all_ranked_crops": ranked
        }

    def _format_location_detail_panel(self, features: Dict[str, Any], preds: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Detailed Location Metadata & Agronomic Attributes Panel."""
        return {
            "soil_properties": {
                "organic_carbon_g_kg": float(features.get("soil_soc_g_kg_0_30cm", 12.0)),
                "ph_level": float(features.get("soil_ph_h2o_0_30cm", 6.5)),
                "cec_cmol_kg": float(features.get("soil_cec_cmol_kg_0_30cm", 15.0)),
                "texture": {
                    "clay_pct": float(features.get("soil_clay_pct_0_30cm", 30.0)),
                    "sand_pct": float(features.get("soil_sand_pct_0_30cm", 40.0)),
                    "silt_pct": float(features.get("soil_silt_pct_0_30cm", 30.0)),
                }
            },
            "climate_forecast": {
                "precipitation_mm": preds.get("current_month_precipitation", {}).get("value", 120.0),
                "mean_temperature_c": preds.get("current_month_mean_temperature", {}).get("value", 27.5),
                "solar_radiation_mj_m2_day": preds.get("current_month_mean_solar_rad", {}).get("value", 18.5),
            },
            "terrain_and_water": {
                "elevation_m": float(features.get("elevation_m", 15.0)),
                "slope_degrees": float(features.get("slope_degrees", 2.0)),
                "distance_to_surface_water_m": float(features.get("distance_to_surface_water_m", 450.0)),
                "surface_water_occurrence_pct": float(features.get("surface_water_occurrence_pct", 20.0)),
            }
        }

    def _format_risk_alerts(self, preds: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize risk alerts (Flood, Drought, Heat, Erosion, Land Conversion, Water Scarcity)."""
        alerts = []

        # 1. Flood Risk
        flood_val = preds.get("flood_risk_level", {}).get("value", 0)
        flood_sev = "HIGH" if flood_val == 2 else ("MEDIUM" if flood_val == 1 else "LOW")
        if flood_sev in ["HIGH", "MEDIUM"]:
            alerts.append({
                "type": "FLOOD_RISK",
                "severity": flood_sev,
                "title": f"{flood_sev.capitalize()} Flood Risk Detected",
                "description": "High precipitation volume or proximity to water bodies creates localized flood potential.",
                "action": "Ensure surface drainage channels are clear and elevate seed beds."
            })

        # 2. Drought Risk
        drought_val = float(preds.get("drought_risk_score", {}).get("value", 0.2))
        if drought_val > 0.5:
            d_sev = "CRITICAL" if drought_val > 0.75 else "HIGH"
            alerts.append({
                "type": "DROUGHT_RISK",
                "severity": d_sev,
                "title": f"{d_sev.capitalize()} Drought Stress Alert",
                "description": f"Soil moisture deficit index at {drought_val*100:.1f}%.",
                "action": "Schedule supplementary drip or canal irrigation immediately."
            })

        # 3. Heat Stress
        heat_val = preds.get("heat_stress_risk", {}).get("value", 0)
        if heat_val == 1:
            alerts.append({
                "type": "HEAT_STRESS",
                "severity": "HIGH",
                "title": "High Heat Stress Warning",
                "description": "Temperatures exceed optimal growth threshold during daytime hours.",
                "action": "Apply organic mulch to retain soil moisture and reduce root zone heat."
            })

        # 4. Water Scarcity
        ws_val = preds.get("water_scarcity_risk", {}).get("label", "Low")
        if ws_val in ["High", "Medium"]:
            alerts.append({
                "type": "WATER_SCARCITY",
                "severity": ws_val.upper(),
                "title": f"{ws_val} Water Scarcity Risk",
                "description": "Surface water proximity and precipitation levels indicate potential irrigation constraints.",
                "action": "Adopt drought-tolerant seed varieties and water conservation techniques."
            })

        return {
            "alert_count": len(alerts),
            "alerts": alerts
        }

    def _format_user_pattern_analysis(self, preds: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """Provide Agronomic Pattern Analysis, Fertilizer recommendations, and Farming Insights."""
        soc = float(features.get("soil_soc_g_kg_0_30cm", 12.0))
        ph = float(features.get("soil_ph_h2o_0_30cm", 6.5))

        n_level = preds.get("nitrogen_requirement_level", {}).get("label", "Medium")
        p_level = preds.get("phosphorus_requirement_level", {}).get("label", "Medium")
        irr_need = preds.get("irrigation_need", {}).get("label", "Medium")
        opt_month = preds.get("optimal_planting_month", {}).get("label", "May (Monsoon)")

        return {
            "agronomic_pattern_summary": {
                "soil_fertility_status": "High" if soc > 18.0 else ("Moderate" if soc > 10.0 else "Low"),
                "soil_acidity_status": "Optimal" if 6.0 <= ph <= 7.2 else ("Acidic" if ph < 6.0 else "Alkaline"),
                "irrigation_need_status": irr_need,
                "optimal_planting_season": opt_month,
            },
            "fertilizer_recommendation": {
                "nitrogen_requirement": n_level,
                "phosphorus_requirement": p_level,
                "potassium_requirement": "Medium",
                "suggested_mix_kg_ha": "N: 90 kg/ha | P2O5: 45 kg/ha | K2O: 30 kg/ha"
            },
            "actionable_agronomic_advice": [
                f"Optimal planting window begins in {opt_month}.",
                f"Maintain irrigation frequency according to {irr_need} deficit level.",
                "Incorporate organic compost to enhance Cation Exchange Capacity (CEC) and soil structure."
            ]
        }

    def _format_market_infrastructure(self, preds: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """Format Market Integration & Supply Chain features."""
        mkt_score = preds.get("market_integration_score", {}).get("value", 0.75)
        loss_risk = preds.get("post_harvest_loss_risk", {}).get("label", "Medium")
        sc_eff = preds.get("supply_chain_efficiency", {}).get("value", 0.80)
        cold_chain = preds.get("cold_chain_potential", {}).get("value", 0.65)

        return {
            "market_integration_score": mkt_score,
            "post_harvest_loss_risk": loss_risk,
            "supply_chain_efficiency_pct": round(sc_eff * 100, 1),
            "cold_chain_potential_pct": round(cold_chain * 100, 1),
            "transport_accessibility": "Good" if mkt_score > 0.7 else "Moderate"
        }

    def _format_gdp_forecast(self, preds: Dict[str, Any]) -> Dict[str, Any]:
        """Format Agricultural GDP Forecast."""
        gdp_val = preds.get("agricultural_gdp_forecast", {}).get("value", 1250.0)
        yield_val = preds.get("crop_yield_t_ha", {}).get("value", 3.8)

        return {
            "estimated_agricultural_gdp_usd_ha": gdp_val,
            "projected_yield_value_usd_ha": round(gdp_val * 0.85, 2),
            "economic_rating": "High Productivity Zone" if gdp_val > 1500 else "Standard Production Zone"
        }
