#!/usr/bin/env python3
"""
estimator_fallback.py — Domain-Logic Fallback Estimators for 40-Target Pipeline
==============================================================================
Provides mathematical & agronomic heuristic predictions for any target model
that is not yet fully trained or available in `gp_models/` or `models/`.
"""

import numpy as np

def estimate_fallback(target: str, features: dict) -> dict:
    """
    Compute heuristic prediction for a given target based on agronomic feature input.
    Returns dict with 'value', 'label' (if classification), and 'is_fallback': True.
    """
    # Extract common features with safe defaults
    ndvi = float(features.get("ndvi_median_mean", 0.55))
    soc = float(features.get("soil_soc_g_kg_0_30cm", 12.0))
    ph = float(features.get("soil_ph_h2o_0_30cm", 6.2))
    moisture = float(features.get("era5_soil_moisture_m3_m3_mean", 0.28))
    precip = float(features.get("chirps_precipitation_mm_mean", 120.0))
    temp = float(features.get("mean_temperature_c_mean", 27.5))
    water_dist = float(features.get("distance_to_surface_water_m", 500.0))
    road_dist = float(features.get("distance_to_road_km", 5.0))
    pop_density = float(features.get("population_density", 150.0))
    builtup = float(features.get("builtup_fraction", 0.05))

    # --- 1-17 Crop Suitabilities ---
    if target.startswith("crop_suitability_"):
        crop = target.replace("crop_suitability_", "")
        # Heuristic suitability score 0-1
        score = (ndvi * 0.4) + (min(soc / 25.0, 1.0) * 0.3) + (min(moisture / 0.4, 1.0) * 0.3)
        if score > 0.75:
            cat = "excellent"
        elif score > 0.55:
            cat = "good"
        elif score > 0.35:
            cat = "moderate"
        else:
            cat = "poor"
        return {"value": score, "label": cat, "is_fallback": True}

    # --- 18 Crop Health Score ---
    elif target == "crop_health_score":
        val = min(max((ndvi * 0.5) + (min(moisture / 0.35, 1.0) * 0.3) + (min(soc / 20.0, 1.0) * 0.2), 0.0), 1.0)
        return {"value": round(val, 3), "label": f"{val*100:.1f}%", "is_fallback": True}

    # --- 19 Crop Yield t/ha ---
    elif target == "crop_yield_t_ha":
        base_yield = 3.8
        multiplier = (ndvi / 0.6) * (min(soc / 15.0, 1.2)) * (min(moisture / 0.25, 1.1))
        val = round(base_yield * multiplier, 2)
        return {"value": float(np.clip(val, 0.5, 7.5)), "label": f"{val} tons/ha", "is_fallback": True}

    # --- 20 Irrigation Need ---
    elif target == "irrigation_need":
        if moisture < 0.18 and precip < 50:
            cat, val = 2, "High"
        elif moisture < 0.28 or precip < 100:
            cat, val = 1, "Medium"
        else:
            cat, val = 0, "Low"
        return {"value": cat, "label": val, "is_fallback": True}

    # --- 21-23 Climate Forecasts ---
    elif target == "current_month_precipitation":
        return {"value": round(precip, 1), "label": f"{precip:.1f} mm", "is_fallback": True}
    elif target == "current_month_mean_temperature":
        return {"value": round(temp, 1), "label": f"{temp:.1f} °C", "is_fallback": True}
    elif target == "current_month_mean_solar_rad":
        sol = float(features.get("solar_radiation_mj_m2_day_mean", 18.5))
        return {"value": round(sol, 1), "label": f"{sol:.1f} MJ/m²/day", "is_fallback": True}

    # --- 24-26 Environmental Hazards ---
    elif target == "flood_risk_level":
        if precip > 250 or water_dist < 100:
            cat, val = 2, "High"
        elif precip > 150 or water_dist < 300:
            cat, val = 1, "Medium"
        else:
            cat, val = 0, "Low"
        return {"value": cat, "label": val, "is_fallback": True}

    elif target == "drought_risk_score":
        drought = float(np.clip(1.0 - (moisture / 0.4 * 0.6 + precip / 200.0 * 0.4), 0.0, 1.0))
        return {"value": round(drought, 3), "label": f"{drought*100:.1f}%", "is_fallback": True}

    elif target == "heat_stress_risk":
        val = 1 if temp > 34.0 else 0
        return {"value": val, "label": "High" if val == 1 else "Normal", "is_fallback": True}

    # --- 27-30 Management & Soil Risks ---
    elif target == "optimal_planting_month":
        month = 5 if precip > 100 else 11
        return {"value": month, "label": "May (Monsoon)" if month == 5 else "November (Dry Season)", "is_fallback": True}

    elif target == "nitrogen_requirement_level":
        n_req = "High" if soc < 10.0 else ("Medium" if soc < 18.0 else "Low")
        return {"value": 2 if n_req == "High" else (1 if n_req == "Medium" else 0), "label": n_req, "is_fallback": True}

    elif target == "phosphorus_requirement_level":
        p_req = "High" if ph < 5.5 or ph > 7.5 else "Medium"
        return {"value": 2 if p_req == "High" else 1, "label": p_req, "is_fallback": True}

    elif target == "soil_erosion_risk":
        slope = float(features.get("slope_degrees", 3.0))
        risk = "High" if slope > 15.0 else ("Medium" if slope > 6.0 else "Low")
        return {"value": 2 if risk == "High" else (1 if risk == "Medium" else 0), "label": risk, "is_fallback": True}

    # --- 31-34 Market & Infrastructure ---
    elif target == "market_integration_score":
        score = float(np.clip(1.0 - (road_dist / 20.0), 0.1, 1.0))
        return {"value": round(score, 2), "label": f"{score*100:.0f}/100", "is_fallback": True}

    elif target == "post_harvest_loss_risk":
        risk = "High" if road_dist > 15.0 else ("Medium" if road_dist > 5.0 else "Low")
        return {"value": 2 if risk == "High" else (1 if risk == "Medium" else 0), "label": risk, "is_fallback": True}

    elif target == "supply_chain_efficiency":
        eff = float(np.clip(0.9 - (road_dist * 0.03), 0.2, 0.95))
        return {"value": round(eff, 2), "label": f"{eff*100:.0f}%", "is_fallback": True}

    elif target == "cold_chain_potential":
        pot = float(np.clip((builtup * 5.0) + (pop_density / 500.0), 0.1, 1.0))
        return {"value": round(pot, 2), "label": f"{pot*100:.0f}%", "is_fallback": True}

    # --- 35-36 Urban & Land Use Change ---
    elif target == "agricultural_land_conversion_risk":
        risk = "High" if builtup > 0.15 or pop_density > 400 else ("Medium" if builtup > 0.05 else "Low")
        return {"value": 2 if risk == "High" else (1 if risk == "Medium" else 0), "label": risk, "is_fallback": True}

    elif target == "urban_encroachment_risk":
        risk = "High" if builtup > 0.20 else ("Medium" if builtup > 0.08 else "Low")
        return {"value": 2 if risk == "High" else (1 if risk == "Medium" else 0), "label": risk, "is_fallback": True}

    # --- 37-39 Water Access & Scarcity ---
    elif target == "irrigation_potential":
        pot = float(np.clip(1.0 - (water_dist / 2000.0), 0.05, 1.0))
        return {"value": round(pot, 2), "label": f"{pot*100:.0f}%", "is_fallback": True}

    elif target == "surface_water_occurrence":
        sw = float(features.get("surface_water_occurrence_pct", 15.0)) / 100.0
        return {"value": round(sw, 2), "label": f"{sw*100:.1f}%", "is_fallback": True}

    elif target == "water_scarcity_risk":
        scarcity = "High" if moisture < 0.15 and water_dist > 1000 else ("Medium" if moisture < 0.25 else "Low")
        return {"value": 2 if scarcity == "High" else (1 if scarcity == "Medium" else 0), "label": scarcity, "is_fallback": True}

    # --- 40 Economic Output ---
    elif target == "agricultural_gdp_forecast":
        gdp = round(1250.0 * (ndvi / 0.5) * (soc / 12.0), 2)
        return {"value": gdp, "label": f"${gdp:,.2f}/ha/yr", "is_fallback": True}

    # Generic Fallback
    return {"value": 0.5, "label": "Moderate", "is_fallback": True}
