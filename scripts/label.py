#!/usr/bin/env python3
"""
label.py — Standalone Labeling Script
======================================
Reads merged/processed CSV files from data/processed/ and applies
detailed, multi-feature labeling rules to generate all 40 prediction columns.

Output: same files updated in-place with the 40 label columns added.

Prediction columns generated (40 total)
---------------------------------------
 1-17 : crop_suitability_<crop>       (excellent / good / moderate / poor)
        Crops: monsoon_rice, dry_season_rice, maize, sugarcane, cassava,
               durian, mangosteen, longan, mango, chili, tomato, black_gram,
               green_gram, pigeon_pea, groundnut, sesame, rubber
18    : crop_health_score              (float 0.0 – 1.0)
19    : crop_yield_t_ha               (float tons/ha)
20    : irrigation_need               (0 = low, 1 = medium, 2 = high)
21    : current_month_precipitation_mm (float – pass-through)
22    : current_month_mean_temperature_c (float – pass-through)
23    : current_month_solar_rad_mj_m2_day (float – pass-through)
24    : flood_risk_level               (0 = low, 1 = medium, 2 = high)
25    : drought_risk_score             (float 0.0 – 1.0)
26    : heat_stress_risk               (0 = low, 1 = medium, 2 = high)
27    : optimal_planting_month         (1 – 12)
28    : nitrogen_requirement_level     (0 = low, 1 = medium, 2 = high)
29    : phosphorus_requirement_level   (0 = low, 1 = medium, 2 = high)
30    : soil_erosion_risk              (0 = low, 1 = medium, 2 = high)
31    : market_integration_score       (float 0.0 – 1.0)
32    : post_harvest_loss_risk         (float 0.0 – 1.0)
33    : supply_chain_efficiency        (float 0.0 – 1.0)
34    : cold_chain_potential           (float 0.0 – 1.0)
35    : agricultural_land_conversion_risk (float 0.0 – 1.0)
36    : urban_encroachment_risk        (float 0.0 – 1.0)
37    : irrigation_potential           (float 0.0 – 1.0)
38    : surface_water_occurrence       (float 0.0 – 1.0)
39    : water_scarcity_risk            (float 0.0 – 1.0)
40    : agricultural_gdp_forecast      (float 0.0 – 1.0)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _safe(df: pd.DataFrame, col: str, default=np.nan):
    """Return series or a default-filled series if column is missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


# ---------------------------------------------------------------------------
# Individual scoring sub-functions (return 0.0 – 1.0 scores)
# ---------------------------------------------------------------------------

def _precip_score(df):
    """Score based on annual mean precipitation and its CV."""
    rain_monthly = _safe(df, "chirps_precipitation_mm_mean", 0)
    rain_annual = rain_monthly * 12.0  # Convert monthly mean mm to annual equivalent mm
    cv   = _safe(df, "chirps_precipitation_mm_cv",   1)  # lower = more reliable
    opt  = 1400.0
    spread = 600.0
    base = np.exp(-0.5 * ((rain_annual - opt) / spread) ** 2)
    rel_pen = np.clip(1.0 - cv * 0.5, 0.3, 1.0)
    return np.clip(base * rel_pen, 0, 1)


def _soil_moisture_score(df):
    """Score based on mean soil moisture and its minimum."""
    sm_mean = _safe(df, "era5_soil_moisture_m3_m3_mean", 0.15)
    sm_min  = _safe(df, "era5_soil_moisture_m3_m3_min",  0.05)
    sm_cv   = _safe(df, "era5_soil_moisture_m3_m3_cv",   0.5)
    opt_mean = 0.30
    base = np.clip(1.0 - np.abs(sm_mean - opt_mean) / 0.3, 0, 1)
    drought_pen = np.clip(sm_min / 0.15, 0, 1)
    cv_pen = np.clip(1.0 - sm_cv * 0.4, 0.3, 1.0)
    return np.clip(base * drought_pen * cv_pen, 0, 1)


def _temperature_score(df, t_opt=27.0, t_spread=8.0):
    """Generic temperature score – override opt/spread per crop."""
    t_mean = _safe(df, "mean_temperature_c_mean",  25)
    t_max  = _safe(df, "mean_temperature_c_max",   35)
    t_min  = _safe(df, "mean_temperature_c_min",   15)
    t_range= _safe(df, "mean_temperature_c_range",  0)
    base = np.exp(-0.5 * ((t_mean - t_opt) / t_spread) ** 2)
    heat_pen = np.where(t_max > 38, np.clip(1.0 - (t_max - 38) / 10.0, 0.1, 1.0), 1.0)
    frost_pen = np.where(t_min < 10, np.clip((t_min - 5) / 5.0, 0.1, 1.0), 1.0)
    range_pen = np.clip(1.0 - (t_range - 10) / 20.0, 0.5, 1.0)
    return np.clip(base * heat_pen * frost_pen * range_pen, 0, 1)


def _ndvi_score(df):
    """Score based on NDVI mean, max and growing-season mean."""
    ndvi_mean = _safe(df, "ndvi_median_mean",               0.3)
    ndvi_max  = _safe(df, "ndvi_median_max",                0.5)
    ndvi_gs   = _safe(df, "ndvi_median_growing_season_mean",0.4)
    base   = np.clip(ndvi_mean / 0.6, 0, 1)
    gs_s   = np.clip(ndvi_gs / 0.65, 0, 1)
    peak_s = np.clip(ndvi_max / 0.75, 0, 1)
    return np.clip((base * 0.4 + gs_s * 0.4 + peak_s * 0.2), 0, 1)


def _ndwi_score(df):
    """Score water availability from NDWI."""
    ndwi_mean = _safe(df, "ndwi_mcf_median_mean", -0.2)
    ndwi_max  = _safe(df, "ndwi_mcf_median_max",   0.0)
    base = np.clip((ndwi_mean + 0.3) / 0.6, 0, 1)
    peak = np.clip((ndwi_max  + 0.2) / 0.5, 0, 1)
    return np.clip(base * 0.6 + peak * 0.4, 0, 1)


def _soil_fertility_score(df):
    """Score based on SOC, CEC, pH, clay, sand, silt."""
    soc  = _safe(df, "soil_soc_g_kg_0_30cm",    20)
    cec  = _safe(df, "soil_cec_cmol_kg_0_30cm",  20)
    ph   = _safe(df, "soil_ph_h2o_0_30cm",       6.5)
    clay = _safe(df, "soil_clay_pct_0_30cm",      25)
    sand = _safe(df, "soil_sand_pct_0_30cm",      40)
    silt = _safe(df, "soil_silt_pct_0_30cm",      35)

    soc_s  = np.clip(soc / 40.0, 0, 1)
    cec_s  = np.clip(cec / 35.0, 0, 1)
    ph_s   = np.exp(-0.5 * ((ph - 6.5) / 0.8) ** 2)
    clay_s = np.exp(-0.5 * ((clay - 25) / 15) ** 2)
    silt_s = np.clip(silt / 40.0, 0, 1)
    sand_pen = np.clip(1.0 - (sand - 50) / 50, 0.1, 1.0)

    return np.clip(
        soc_s * 0.30 + cec_s * 0.25 + ph_s * 0.20 +
        clay_s * 0.10 + silt_s * 0.10 + sand_pen * 0.05,
        0, 1
    )


def _terrain_score(df):
    """Score based on elevation, slope and aspect."""
    elev  = _safe(df, "elevation_m",    50)
    slope = _safe(df, "slope_degrees",   3)
    aspect= _safe(df, "aspect_degrees", 180)

    elev_s  = np.where(elev < 300, 1.0, np.clip(1.0 - (elev - 300) / 700, 0.2, 1.0))
    slope_s = np.clip(1.0 - slope / 20.0, 0.1, 1.0)
    aspect_s = 0.75 + 0.25 * np.cos(np.radians(aspect - 180))

    return np.clip(elev_s * 0.4 + slope_s * 0.4 + aspect_s * 0.2, 0, 1)


def _water_proximity_score(df):
    """Proximity to surface water / rivers — irrigation access."""
    dist = _safe(df, "distance_to_surface_water_m", 2000)
    if "distance_to_river_km" in df.columns:
        river_dist_m = _safe(df, "distance_to_river_km", 2.0) * 1000.0
        dist = np.minimum(dist, river_dist_m)
    occ  = _safe(df, "surface_water_occurrence_pct",   50)
    seas = _safe(df, "surface_water_seasonality_months", 6)

    dist_s = np.clip(1.0 - dist / 5000.0, 0.1, 1.0)
    occ_s  = np.clip(occ / 90.0, 0, 1)
    seas_s = np.clip(seas / 10.0, 0, 1)
    return np.clip(dist_s * 0.5 + occ_s * 0.3 + seas_s * 0.2, 0, 1)


def _solar_score(df):
    """Score solar radiation suitability."""
    rad_mean = _safe(df, "solar_radiation_mj_m2_day_mean", 18)
    rad_max  = _safe(df, "solar_radiation_mj_m2_day_max",  22)
    base = np.clip(rad_mean / 20.0, 0, 1)
    stress_pen = np.where(rad_max > 25, np.clip(1.0 - (rad_max - 25) / 10, 0.4, 1.0), 1.0)
    return np.clip(base * stress_pen, 0, 1)


def _radar_score(df):
    """SAR backscatter – surface moisture / vegetation structure."""
    vh = _safe(df, "s1_vh_db_median_mean", -15)
    vv = _safe(df, "s1_vv_db_median_mean", -10)
    vh_s = np.clip((vh + 20) / 12.0, 0, 1)
    vv_s = np.clip((vv + 16) / 12.0, 0, 1)
    return np.clip((vh_s + vv_s) / 2, 0, 1)


# ---------------------------------------------------------------------------
# Predictions 1-17: Composite Suitability Profiles
# ---------------------------------------------------------------------------
CROP_PROFILES = {
    "monsoon_rice":    (25, 6, 1800, 600,  [0.18, 0.20, 0.12, 0.12, 0.10, 0.10, 0.06, 0.08, 0.04]),
    "dry_season_rice": (26, 6, 1400, 500,  [0.12, 0.20, 0.12, 0.12, 0.10, 0.10, 0.06, 0.12, 0.06]),
    "maize":           (25, 6, 800,  400,  [0.16, 0.14, 0.14, 0.12, 0.08, 0.14, 0.10, 0.06, 0.06]),
    "sugarcane":       (27, 6, 1500, 500,  [0.16, 0.16, 0.13, 0.12, 0.08, 0.12, 0.09, 0.08, 0.06]),
    "cassava":         (26, 7, 1000, 500,  [0.14, 0.12, 0.14, 0.12, 0.08, 0.14, 0.12, 0.07, 0.07]),
    "durian":          (27, 5, 1800, 600,  [0.15, 0.15, 0.14, 0.13, 0.09, 0.12, 0.09, 0.07, 0.06]),
    "mangosteen":      (27, 5, 2000, 600,  [0.14, 0.16, 0.14, 0.13, 0.09, 0.12, 0.09, 0.07, 0.06]),
    "longan":          (25, 6, 1200, 500,  [0.14, 0.13, 0.14, 0.12, 0.09, 0.13, 0.10, 0.08, 0.07]),
    "mango":           (26, 7, 1000, 500,  [0.13, 0.12, 0.15, 0.12, 0.09, 0.13, 0.12, 0.08, 0.06]),
    "chili":           (25, 6, 900,  400,  [0.14, 0.13, 0.14, 0.12, 0.08, 0.15, 0.10, 0.08, 0.06]),
    "tomato":          (22, 5, 800,  400,  [0.13, 0.13, 0.16, 0.12, 0.08, 0.15, 0.10, 0.07, 0.06]),
    "black_gram":      (26, 6, 900,  400,  [0.14, 0.14, 0.14, 0.12, 0.08, 0.16, 0.10, 0.06, 0.06]),
    "green_gram":      (26, 6, 800,  350,  [0.14, 0.14, 0.14, 0.12, 0.08, 0.16, 0.10, 0.06, 0.06]),
    "pigeon_pea":      (27, 7, 750,  400,  [0.12, 0.12, 0.16, 0.12, 0.08, 0.16, 0.12, 0.06, 0.06]),
    "groundnut":       (26, 6, 850,  400,  [0.14, 0.13, 0.14, 0.12, 0.08, 0.17, 0.10, 0.06, 0.06]),
    "sesame":          (28, 6, 700,  350,  [0.12, 0.11, 0.16, 0.12, 0.07, 0.18, 0.12, 0.06, 0.06]),
    "rubber":          (27, 5, 2200, 600,  [0.18, 0.18, 0.14, 0.12, 0.08, 0.12, 0.08, 0.06, 0.04]),
}

SUITABILITY_LABELS = ["poor", "moderate", "good", "excellent"]

def _suitability_score(df, crop_key: str) -> pd.Series:
    """Compute composite suitability score (0-1) for a crop."""
    t_opt, t_sp, r_opt, r_sp, w = CROP_PROFILES[crop_key]

    rain_monthly = _safe(df, "chirps_precipitation_mm_mean", 0)
    rain_annual = rain_monthly * 12.0
    cv   = _safe(df, "chirps_precipitation_mm_cv",   1)
    precip_s = np.exp(-0.5 * ((rain_annual - r_opt) / r_sp) ** 2) * np.clip(1 - cv * 0.5, 0.3, 1)

    components = [
        np.clip(precip_s, 0, 1),
        _soil_moisture_score(df),
        _temperature_score(df, t_opt=t_opt, t_spread=t_sp),
        _ndvi_score(df),
        _ndwi_score(df),
        _soil_fertility_score(df),
        _terrain_score(df),
        _water_proximity_score(df),
        _solar_score(df),
    ]
    score = sum(c * w[i] for i, c in enumerate(components))

    elev = _safe(df, "elevation_m", 100)
    slope = _safe(df, "slope_degrees", 5)
    water_occ = _safe(df, "surface_water_occurrence_pct", 0)
    sm = _safe(df, "era5_soil_moisture_m3_m3_mean", 0.25)

    rice_pct = _safe(df, "crop_area_pct_monsoon_rice", 20.0) + _safe(df, "crop_area_pct_dry_season_rice", 10.0)
    if crop_key in ["monsoon_rice", "dry_season_rice"]:
        paddy_condition = (elev < 60) & (slope < 3.0) & ((water_occ > 15) | (sm > 0.27))
        rice_share_boost = np.where(rice_pct > 50.0, 1.15, 1.0)
        score = np.where(paddy_condition, score * 1.20, score) * rice_share_boost

    elif crop_key in ["mango", "durian", "longan", "mangosteen", "cassava", "tomato", "sesame", "pigeon_pea"]:
        waterlog_condition = (water_occ > 25) | ((elev < 30) & (slope < 1.5) & (sm > 0.30))
        rice_comp_pen = np.where(rice_pct > 70.0, 0.90, 1.0)
        score = np.where(waterlog_condition, score * 0.75, score) * rice_comp_pen

    return np.clip(score, 0, 1)


def _score_to_label(score: pd.Series) -> pd.Series:
    """Map score to categorical label."""
    return pd.cut(
        score,
        bins=[-0.001, 0.40, 0.60, 0.80, 1.001],
        labels=["poor", "moderate", "good", "excellent"],
        right=True,
    ).astype(str)


# ---------------------------------------------------------------------------
# Prediction 18: Crop Health Score
# ---------------------------------------------------------------------------
def compute_health_score(df: pd.DataFrame) -> pd.Series:
    """Prediction 18: Crop Health Score (float 0.0 – 1.0)."""
    ndvi_s  = _ndvi_score(df)
    sm_s    = _soil_moisture_score(df)
    temp_s  = _temperature_score(df, t_opt=27, t_spread=7)
    ndwi_s  = _ndwi_score(df)
    soil_s  = _soil_fertility_score(df)
    solar_s = _solar_score(df)
    radar_s = _radar_score(df)

    score = (
        ndvi_s  * 0.28 +
        sm_s    * 0.20 +
        temp_s  * 0.18 +
        ndwi_s  * 0.13 +
        soil_s  * 0.12 +
        solar_s * 0.05 +
        radar_s * 0.04
    )
    return np.clip(score, 0, 1).round(4)


# ---------------------------------------------------------------------------
# Prediction 19: Crop Yield Prediction (tons/ha)
# ---------------------------------------------------------------------------
def compute_yield(df: pd.DataFrame) -> pd.Series:
    """Prediction 19: Crop Yield Prediction (tons/ha)."""
    BASE_YIELD = 5.5
    precip_s = _precip_score(df)
    sm_s     = _soil_moisture_score(df)
    temp_s   = _temperature_score(df)
    ndvi_s   = _ndvi_score(df)
    soil_s   = _soil_fertility_score(df)
    water_s  = _water_proximity_score(df)

    limiting = pd.concat([precip_s, sm_s, temp_s, soil_s], axis=1).min(axis=1)

    composite = (
        limiting  * 0.35 +
        ndvi_s    * 0.25 +
        sm_s      * 0.15 +
        soil_s    * 0.15 +
        water_s   * 0.10
    )
    yield_val = BASE_YIELD * np.clip(composite, 0, 1)
    slope = _safe(df, "slope_degrees", 3)
    yield_val *= np.clip(1.0 - (slope - 5) / 30.0, 0.6, 1.0)
    return np.clip(yield_val, 0.5, 6.0).round(3)


# ---------------------------------------------------------------------------
# Prediction 20: Irrigation Need (0 = Low, 1 = Medium, 2 = High)
# ---------------------------------------------------------------------------
def compute_irrigation_need(df: pd.DataFrame) -> pd.Series:
    """Prediction 20: Irrigation Need Level (0/1/2)."""
    rain      = _safe(df, "chirps_precipitation_mm_mean",   100)
    rain_min  = _safe(df, "chirps_precipitation_mm_min",     20)
    rain_cv   = _safe(df, "chirps_precipitation_mm_cv",       0.5)
    sm_mean   = _safe(df, "era5_soil_moisture_m3_m3_mean",   0.2)
    sm_min    = _safe(df, "era5_soil_moisture_m3_m3_min",    0.08)
    ndwi      = _safe(df, "ndwi_mcf_median_mean",            -0.2)
    sand      = _safe(df, "soil_sand_pct_0_30cm",             40)
    clay      = _safe(df, "soil_clay_pct_0_30cm",             25)
    dist_w    = _safe(df, "distance_to_surface_water_m",    2000)
    occ       = _safe(df, "surface_water_occurrence_pct",     50)

    demand = pd.Series(0.0, index=df.index)
    demand += np.where(rain < 50,  0.30, np.where(rain < 100, 0.15, 0.0))
    demand += np.where(rain_min < 10, 0.15, np.where(rain_min < 30, 0.07, 0.0))
    demand += np.clip(rain_cv * 0.15, 0, 0.10)
    demand += np.where(sm_mean < 0.10, 0.25, np.where(sm_mean < 0.18, 0.12, 0.0))
    demand += np.where(sm_min  < 0.06, 0.15, np.where(sm_min  < 0.12, 0.07, 0.0))
    demand += np.where(ndwi < -0.3, 0.15, np.where(ndwi < 0, 0.05, 0.0))
    demand += np.clip((sand - 40) / 60.0 * 0.10, 0, 0.10)
    demand -= np.clip((clay - 30) / 40.0 * 0.05, 0, 0.05)
    demand += np.where(dist_w > 3000, 0.05, 0.0)
    demand -= np.clip(occ / 90 * 0.05, 0, 0.05)

    need = pd.Series(0, index=df.index, dtype=int)
    need[demand >= 0.35] = 2
    need[(demand >= 0.18) & (demand < 0.35)] = 1
    return need


# ---------------------------------------------------------------------------
# Predictions 24-30: Climate Risk & Farm Management Targets
# ---------------------------------------------------------------------------
def compute_flood_risk(df: pd.DataFrame) -> pd.Series:
    """Prediction 24: Flood Risk Level (0=Low, 1=Medium, 2=High)."""
    water_occ = _safe(df, "surface_water_occurrence_pct", 0)
    elev      = _safe(df, "elevation_m", 100)
    slope     = _safe(df, "slope_degrees", 5)
    vv        = _safe(df, "s1_vv_db_median_mean", -12)

    risk = pd.Series(0, index=df.index, dtype=int)
    high_cond = (water_occ > 50) | ((elev < 15) & (slope < 1.0) & (vv > -9))
    med_cond  = (water_occ > 20) | ((elev < 35) & (slope < 2.5))
    risk[med_cond]  = 1
    risk[high_cond] = 2
    return risk


def compute_drought_risk(df: pd.DataFrame) -> pd.Series:
    """Prediction 25: Drought Risk Score (float 0.0 – 1.0)."""
    sm_min  = _safe(df, "era5_soil_moisture_m3_m3_min", 0.20)
    sm_cv   = _safe(df, "era5_soil_moisture_m3_m3_cv", 0.3)
    rain_cv = _safe(df, "chirps_precipitation_mm_cv", 0.5)
    ndwi    = _safe(df, "ndwi_mcf_median_mean", -0.2)

    sm_deficit = np.clip((0.25 - sm_min) / 0.25, 0, 1)
    water_def  = np.clip((-0.2 - ndwi) / 0.5, 0, 1)
    risk_score = sm_deficit * 0.4 + water_def * 0.3 + sm_cv * 0.15 + rain_cv * 0.15
    return np.clip(risk_score, 0, 1).round(4)


def compute_heat_stress_risk(df: pd.DataFrame) -> pd.Series:
    """Prediction 26: Heat Stress Risk Level (0=Low, 1=Medium, 2=High)."""
    t_max  = _safe(df, "mean_temperature_c_max", 30)
    t_mean = _safe(df, "mean_temperature_c_mean", 25)

    risk = pd.Series(0, index=df.index, dtype=int)
    risk[(t_max > 34.0) | (t_mean > 29.0)] = 1
    risk[(t_max > 38.0) | (t_mean > 32.0)] = 2
    return risk


def compute_optimal_planting_month(df: pd.DataFrame) -> pd.Series:
    """Prediction 27: Recommended Optimal Sowing Month (1–12)."""
    region = df.get("region", pd.Series("", index=df.index)).astype(str).str.lower()
    t_mean = _safe(df, "mean_temperature_c_mean", 25)

    month = pd.Series(6, index=df.index, dtype=int)
    month[region == "ayeyawaddy"] = 5
    month[region == "bago"]       = 5
    month[region == "yangon"]     = 5
    month[(region == "magway") & (t_mean > 28)] = 7
    return month


def compute_nitrogen_requirement(df: pd.DataFrame) -> pd.Series:
    """Prediction 28: Nitrogen Requirement Level (0=Low, 1=Med, 2=High)."""
    soc = _safe(df, "soil_soc_g_kg_0_30cm", 25)
    req = pd.Series(1, index=df.index, dtype=int)
    req[soc < 15.0] = 2
    req[soc > 35.0] = 0
    return req


def compute_phosphorus_requirement(df: pd.DataFrame) -> pd.Series:
    """Prediction 29: Phosphorus Requirement Level (0=Low, 1=Med, 2=High)."""
    ph  = _safe(df, "soil_ph_h2o_0_30cm", 6.5)
    cec = _safe(df, "soil_cec_cmol_kg_0_30cm", 20)
    req = pd.Series(1, index=df.index, dtype=int)
    req[(ph < 5.5) | (ph > 7.5) | (cec < 12.0)] = 2
    req[(ph >= 6.0) & (ph <= 7.0) & (cec >= 25.0)] = 0
    return req


def compute_soil_erosion_risk(df: pd.DataFrame) -> pd.Series:
    """Prediction 30: Soil Erosion Risk Level (0=Low, 1=Med, 2=High)."""
    slope = _safe(df, "slope_degrees", 2)
    rain  = _safe(df, "chirps_precipitation_mm_max", 100)

    risk = pd.Series(0, index=df.index, dtype=int)
    risk[(slope > 4.0) | (rain > 300)] = 1
    risk[(slope > 10.0) | ((slope > 5.0) & (rain > 400))] = 2
    return risk


# ---------------------------------------------------------------------------
# Predictions 31-40: Market, Supply Chain, Urbanization, Water & GDP Models
# ---------------------------------------------------------------------------
def compute_market_integration_score(df: pd.DataFrame) -> pd.Series:
    """Prediction 31: Market Integration Score (float 0.0 - 1.0)."""
    d_road = _safe(df, "distance_to_road_km", 10.0)
    d_rail = _safe(df, "distance_to_railway_km", 20.0)
    r_dens = _safe(df, "road_density_km_per_sqkm", 0.5)
    rw_dens = _safe(df, "railway_density_km_per_sqkm", 0.1)

    s_road_dist = np.clip(1.0 - d_road / 30.0, 0, 1)
    s_rail_dist = np.clip(1.0 - d_rail / 50.0, 0, 1)
    s_road_dens = np.clip(r_dens / 2.0, 0, 1)
    s_rail_dens = np.clip(rw_dens / 0.5, 0, 1)

    score = s_road_dist * 0.4 + s_road_dens * 0.3 + s_rail_dist * 0.2 + s_rail_dens * 0.1
    return np.clip(score, 0, 1).round(4)


def compute_post_harvest_loss_risk(df: pd.DataFrame) -> pd.Series:
    """Prediction 32: Post-Harvest Loss Risk (float 0.0 - 1.0)."""
    d_road = _safe(df, "distance_to_road_km", 10.0)
    r_dens = _safe(df, "road_density_km_per_sqkm", 0.5)
    crop_pct = _safe(df, "cropland_fraction", 0.5)
    d_river = _safe(df, "distance_to_river_km", 5.0)

    isolation_penalty = np.clip(d_road / 25.0, 0, 1) * 0.4 + np.clip(1.0 - r_dens / 1.5, 0, 1) * 0.3
    crop_factor = np.clip(crop_pct, 0, 1) * 0.15
    river_factor = np.clip(d_river / 20.0, 0, 1) * 0.15

    risk = isolation_penalty + crop_factor + river_factor
    return np.clip(risk, 0, 1).round(4)


def compute_supply_chain_efficiency(df: pd.DataFrame) -> pd.Series:
    """Prediction 33: Supply Chain Efficiency (float 0.0 - 1.0)."""
    market_score = compute_market_integration_score(df)
    pop_density = _safe(df, "population_density", 100.0)
    s_pop = np.clip(pop_density / 1000.0, 0.1, 1.0)
    crop_frac = _safe(df, "cropland_fraction", 0.5)

    efficiency = market_score * 0.6 + s_pop * 0.25 + crop_frac * 0.15
    return np.clip(efficiency, 0, 1).round(4)


def compute_cold_chain_potential(df: pd.DataFrame) -> pd.Series:
    """Prediction 34: Cold Chain Potential (float 0.0 - 1.0)."""
    r_dens = _safe(df, "road_density_km_per_sqkm", 0.5)
    urban_f = _safe(df, "urban_fraction", 0.05)
    built_f = _safe(df, "builtup_fraction", 0.05)
    pop_d = _safe(df, "population_density", 100.0)

    s_road = np.clip(r_dens / 2.0, 0, 1)
    s_urban = np.clip((urban_f + built_f) / 0.4, 0, 1)
    s_pop = np.clip(pop_d / 800.0, 0, 1)

    potential = s_road * 0.4 + s_urban * 0.4 + s_pop * 0.2
    return np.clip(potential, 0, 1).round(4)


def compute_agricultural_land_conversion_risk(df: pd.DataFrame) -> pd.Series:
    """Prediction 35: Agricultural Land Conversion Risk (float 0.0 - 1.0)."""
    urban_f = _safe(df, "urban_fraction", 0.05)
    built_f = _safe(df, "builtup_fraction", 0.05)
    pop_d = _safe(df, "population_density", 100.0)
    r_dens = _safe(df, "road_density_km_per_sqkm", 0.5)
    crop_f = _safe(df, "cropland_fraction", 0.5)

    urban_expansion = np.clip((urban_f + built_f) / 0.3, 0, 1)
    pop_pressure = np.clip(pop_d / 1000.0, 0, 1)
    road_access = np.clip(r_dens / 2.0, 0, 1)

    risk = (urban_expansion * 0.4 + pop_pressure * 0.3 + road_access * 0.3) * np.clip(crop_f * 1.5, 0.2, 1.0)
    return np.clip(risk, 0, 1).round(4)


def compute_urban_encroachment_risk(df: pd.DataFrame) -> pd.Series:
    """Prediction 36: Urban Encroachment Risk (float 0.0 - 1.0)."""
    urban_f = _safe(df, "urban_fraction", 0.05)
    built_f = _safe(df, "builtup_fraction", 0.05)
    pop_d = _safe(df, "population_density", 100.0)
    d_road = _safe(df, "distance_to_road_km", 5.0)

    urban_prox = np.clip((urban_f * 2.0 + built_f * 2.0), 0, 1)
    pop_prox = np.clip(pop_d / 1200.0, 0, 1)
    road_prox = np.clip(1.0 - d_road / 20.0, 0, 1)

    risk = urban_prox * 0.45 + pop_prox * 0.35 + road_prox * 0.20
    return np.clip(risk, 0, 1).round(4)


def compute_irrigation_potential(df: pd.DataFrame) -> pd.Series:
    """Prediction 37: Irrigation Potential (float 0.0 - 1.0)."""
    d_river = _safe(df, "distance_to_river_km", 5.0)
    r_dens = _safe(df, "river_density_km_per_sqkm", 0.3)
    water_occ = _safe(df, "surface_water_occurrence_pct", 20.0)
    perm_water = _safe(df, "permanent_water_fraction", 0.05)

    s_river_dist = np.clip(1.0 - d_river / 15.0, 0, 1)
    s_river_dens = np.clip(r_dens / 1.0, 0, 1)
    s_occ = np.clip(water_occ / 80.0, 0, 1)
    s_perm = np.clip(perm_water / 0.3, 0, 1)

    potential = s_river_dist * 0.35 + s_river_dens * 0.25 + s_occ * 0.25 + s_perm * 0.15
    return np.clip(potential, 0, 1).round(4)


def compute_surface_water_occurrence_target(df: pd.DataFrame) -> pd.Series:
    """Prediction 38: Surface Water Occurrence Score (float 0.0 - 1.0)."""
    water_occ = _safe(df, "surface_water_occurrence_pct", 0.0)
    perm_water = _safe(df, "permanent_water_fraction", 0.0)
    score = np.clip(water_occ / 100.0 * 0.7 + perm_water * 0.3, 0, 1)
    return np.clip(score, 0, 1).round(4)


def compute_water_scarcity_risk(df: pd.DataFrame) -> pd.Series:
    """Prediction 39: Water Scarcity Risk (float 0.0 - 1.0)."""
    r_dens = _safe(df, "river_density_km_per_sqkm", 0.3)
    precip_mean = _safe(df, "chirps_precipitation_mm_mean", 100.0)
    sm_mean = _safe(df, "era5_soil_moisture_m3_m3_mean", 0.2)

    lack_river = np.clip(1.0 - r_dens / 1.0, 0, 1)
    lack_rain = np.clip(1.0 - precip_mean / 180.0, 0, 1)
    lack_sm = np.clip((0.35 - sm_mean) / 0.35, 0, 1)

    risk = lack_river * 0.3 + lack_rain * 0.4 + lack_sm * 0.3
    return np.clip(risk, 0, 1).round(4)


def compute_agricultural_gdp_forecast(df: pd.DataFrame) -> pd.Series:
    """Prediction 40: Agricultural GDP Forecast Index (float 0.0 - 1.0)."""
    yield_val = compute_yield(df)
    market_score = compute_market_integration_score(df)
    crop_frac = _safe(df, "cropland_fraction", 0.5)
    health_score = compute_health_score(df)

    norm_yield = np.clip(yield_val / 6.0, 0, 1)
    gdp_index = (norm_yield * 0.4 + market_score * 0.3 + crop_frac * 0.2 + health_score * 0.1)
    return np.clip(gdp_index, 0, 1).round(4)


REGIONAL_CROP_PCT = {
    "ayeyawaddy": {
        "monsoon_rice": 62.0, "dry_season_rice": 24.0, "black_gram": 7.0, "green_gram": 2.5,
        "maize": 1.0, "groundnut": 1.0, "chili": 0.8, "sesame": 0.5, "cassava": 0.3,
        "sugarcane": 0.3, "mango": 0.2, "pigeon_pea": 0.1, "tomato": 0.1, "durian": 0.05,
        "mangosteen": 0.05, "longan": 0.05, "rubber": 0.05
    },
    "bago": {
        "monsoon_rice": 58.0, "dry_season_rice": 18.0, "black_gram": 10.0, "green_gram": 5.0,
        "sugarcane": 3.0, "groundnut": 1.5, "sesame": 1.5, "maize": 1.2, "rubber": 0.8,
        "mango": 0.3, "chili": 0.3, "cassava": 0.2, "pigeon_pea": 0.1, "tomato": 0.05,
        "durian": 0.02, "mangosteen": 0.02, "longan": 0.01
    },
    "yangon": {
        "monsoon_rice": 65.0, "dry_season_rice": 15.0, "black_gram": 6.0, "green_gram": 4.0,
        "groundnut": 2.5, "chili": 2.0, "tomato": 1.5, "cassava": 1.2, "mango": 1.0,
        "maize": 0.8, "sesame": 0.4, "sugarcane": 0.3, "rubber": 0.2, "pigeon_pea": 0.05,
        "durian": 0.02, "mangosteen": 0.02, "longan": 0.01
    },
    "magway": {
        "sesame": 32.0, "groundnut": 22.0, "pigeon_pea": 14.0, "monsoon_rice": 11.0,
        "green_gram": 8.0, "maize": 4.5, "dry_season_rice": 3.5, "chili": 1.5,
        "black_gram": 1.5, "tomato": 0.8, "mango": 0.5, "sugarcane": 0.4, "cassava": 0.2,
        "rubber": 0.05, "durian": 0.01, "mangosteen": 0.01, "longan": 0.03
    },
    "mandalay": {
        "sesame": 26.0, "groundnut": 19.0, "pigeon_pea": 16.0, "monsoon_rice": 13.0,
        "green_gram": 9.0, "maize": 6.0, "dry_season_rice": 4.0, "chili": 2.5,
        "tomato": 2.0, "mango": 1.2, "sugarcane": 0.8, "black_gram": 0.4, "cassava": 0.05,
        "rubber": 0.01, "durian": 0.01, "mangosteen": 0.01, "longan": 0.02
    },
    "sagaing": {
        "monsoon_rice": 32.0, "sesame": 18.0, "groundnut": 14.0, "maize": 10.0,
        "pigeon_pea": 9.0, "green_gram": 6.0, "dry_season_rice": 5.0, "black_gram": 2.5,
        "sugarcane": 1.5, "chili": 0.8, "mango": 0.5, "tomato": 0.4, "cassava": 0.2,
        "rubber": 0.05, "durian": 0.01, "mangosteen": 0.01, "longan": 0.03
    }
}


def add_regional_crop_pct_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add crop_area_pct_<crop> feature columns based on regional statistics."""
    region_series = df.get("region", pd.Series("", index=df.index)).astype(str).str.lower()
    crops = list(CROP_PROFILES.keys())
    for crop in crops:
        col = f"crop_area_pct_{crop}"
        pct_values = region_series.map(
            lambda r: REGIONAL_CROP_PCT.get(r, {}).get(crop, 1.0)
        )
        df[col] = pct_values
    return df


def compute_passthrough(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pass-through targets for current month climate features."""
    res = pd.DataFrame(index=df.index)
    res["current_month_precipitation_mm"] = _safe(df, "chirps_precipitation_mm", 0.0)
    res["current_month_mean_temperature_c"] = _safe(df, "mean_temperature_c", 25.0)
    res["current_month_solar_rad_mj_m2_day"] = _safe(df, "solar_radiation_mj_m2_day", 18.0)
    return res


# ---------------------------------------------------------------------------
# Main labeling function
# ---------------------------------------------------------------------------
def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all 40 prediction labels to df and return updated DataFrame."""
    df = df.copy()

    df = add_regional_crop_pct_features(df)

    label_cols_all = [f"crop_suitability_{c}" for c in CROP_PROFILES.keys()] + [
        "crop_health_score", "crop_yield_t_ha", "irrigation_need",
        "current_month_precipitation_mm", "current_month_mean_temperature_c", "current_month_solar_rad_mj_m2_day",
        "flood_risk_level", "drought_risk_score", "heat_stress_risk",
        "optimal_planting_month", "nitrogen_requirement_level", "phosphorus_requirement_level",
        "soil_erosion_risk", "market_integration_score", "post_harvest_loss_risk",
        "supply_chain_efficiency", "cold_chain_potential", "agricultural_land_conversion_risk",
        "urban_encroachment_risk", "irrigation_potential", "surface_water_occurrence",
        "water_scarcity_risk", "agricultural_gdp_forecast"
    ]
    existing = [c for c in label_cols_all if c in df.columns]
    if existing:
        df.drop(columns=existing, inplace=True)

    # 1-17: Crop suitability
    for crop_key in CROP_PROFILES:
        col_name = f"crop_suitability_{crop_key}"
        score = _suitability_score(df, crop_key)
        df[col_name] = _score_to_label(score)

    # 18: Health score
    df["crop_health_score"] = compute_health_score(df)

    # 19: Yield
    df["crop_yield_t_ha"] = compute_yield(df)

    # 20: Irrigation need
    df["irrigation_need"] = compute_irrigation_need(df)

    # 21-23: Pass-through climate
    passthrough = compute_passthrough(df)
    df = pd.concat([df, passthrough], axis=1)

    # 24-30: Risk & Farm Management Targets
    df["flood_risk_level"]             = compute_flood_risk(df)
    df["drought_risk_score"]           = compute_drought_risk(df)
    df["heat_stress_risk"]             = compute_heat_stress_risk(df)
    df["optimal_planting_month"]       = compute_optimal_planting_month(df)
    df["nitrogen_requirement_level"]   = compute_nitrogen_requirement(df)
    df["phosphorus_requirement_level"] = compute_phosphorus_requirement(df)
    df["soil_erosion_risk"]            = compute_soil_erosion_risk(df)

    # 31-40: Market, Supply Chain, Urbanization, Water & GDP Models
    df["market_integration_score"]          = compute_market_integration_score(df)
    df["post_harvest_loss_risk"]            = compute_post_harvest_loss_risk(df)
    df["supply_chain_efficiency"]           = compute_supply_chain_efficiency(df)
    df["cold_chain_potential"]              = compute_cold_chain_potential(df)
    df["agricultural_land_conversion_risk"] = compute_agricultural_land_conversion_risk(df)
    df["urban_encroachment_risk"]           = compute_urban_encroachment_risk(df)
    df["irrigation_potential"]              = compute_irrigation_potential(df)
    df["surface_water_occurrence"]          = compute_surface_water_occurrence_target(df)
    df["water_scarcity_risk"]               = compute_water_scarcity_risk(df)
    df["agricultural_gdp_forecast"]         = compute_agricultural_gdp_forecast(df)

    return df


# ---------------------------------------------------------------------------
# File discovery and processing
# ---------------------------------------------------------------------------
def find_processed_files(root: Path, region_filter: str = None):
    """Yield (path, region, year, month) tuples for every processed CSV."""
    for region_dir in sorted(root.iterdir()):
        if not region_dir.is_dir():
            continue
        if region_filter and region_dir.name != region_filter:
            continue
        for year_dir in sorted(region_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                csv_file = month_dir / "data.csv"
                if csv_file.is_file():
                    yield csv_file, region_dir.name, year_dir.name, month_dir.name


def process_file(csv_path: Path):
    """Load CSV, label it, overwrite with labeled version."""
    print(f"  Labeling: {csv_path.relative_to(PROJECT_ROOT)}", end=" ... ")
    df = pd.read_csv(csv_path, low_memory=False)
    df = label_dataframe(df)
    df.to_csv(csv_path, index=False)
    print(f"Done  [{len(df)} rows, {df.shape[1]} cols]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Apply tight multi-feature labeling to processed CSV files for all 40 targets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--region", "-r",
        default=None,
        help="Process only this region (e.g., ayeyawaddy). Default: all regions.",
    )
    parser.add_argument(
        "--processed-root", "-p",
        default=str(PROCESSED_ROOT),
        help=f"Root directory of processed data. Default: {PROCESSED_ROOT}",
    )
    args = parser.parse_args()

    root = Path(args.processed_root).resolve()
    if not root.is_dir():
        print(f"[ERROR] Processed data directory not found: {root}")
        sys.exit(1)

    files = list(find_processed_files(root, region_filter=args.region))
    if not files:
        print(f"[WARN] No processed CSV files found under: {root}")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f" Myanmar Agricultural ML — Labeling Pipeline (40 Targets)")
    print(f" Found {len(files)} file(s) to label")
    print(f"{'='*60}\n")

    for csv_path, region, year, month in files:
        process_file(csv_path)

    print(f"\n{'='*60}")
    print(f" All files labeled successfully with 40 target predictions!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
