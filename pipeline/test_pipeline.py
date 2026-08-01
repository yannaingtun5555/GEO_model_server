#!/usr/bin/env python3
"""
test_pipeline.py — Verification & Test Suite for Model Pipeline
=================================================================
Tests model loading, 40-target evaluation, fallback estimation, JSON serialization,
and verifies all required Web Backend feature modules.
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import ModelPipeline, TARGET_DEFINITIONS, CROPS


def main():
    print("==========================================================================")
    print("             MODEL PIPELINE & BE JSON INTEGRATION TESTER                  ")
    print("==========================================================================")

    # 1. Initialize Pipeline
    print("\n[STEP 1] Initializing ModelPipeline...")
    t0 = time.time()
    pipeline = ModelPipeline()
    init_time = (time.time() - t0) * 1000

    print(f"   ↳ Pipeline initialized in {init_time:.2f} ms")
    print(f"   ↳ ML Models Loaded     : {len(pipeline.loaded_targets)} / {len(TARGET_DEFINITIONS)}")
    print(f"   ↳ Fallback Estimators  : {len(pipeline.fallback_targets)} / {len(TARGET_DEFINITIONS)}")

    print("\n   Loaded Target Models:")
    for t in sorted(pipeline.loaded_targets):
        print(f"     • {t}")

    # 2. Test Input Dictionary Prediction
    print("\n[STEP 2] Running Pipeline on Input Feature Dictionary...")
    sample_features = {
        "latitude": 16.8661,
        "longitude": 96.1951,
        "region": "Ayeyawaddy",
        "elevation_m": 12.5,
        "slope_degrees": 1.8,
        "distance_to_surface_water_m": 350.0,
        "soil_soc_g_kg_0_30cm": 15.2,
        "soil_ph_h2o_0_30cm": 6.4,
        "soil_cec_cmol_kg_0_30cm": 18.0,
        "soil_clay_pct_0_30cm": 35.0,
        "soil_sand_pct_0_30cm": 30.0,
        "soil_silt_pct_0_30cm": 35.0,
        "chirps_precipitation_mm_mean": 145.0,
        "era5_soil_moisture_m3_m3_mean": 0.32,
        "mean_temperature_c_mean": 28.2,
        "ndvi_median_mean": 0.68,
        "ndwi_mcf_median_mean": 0.22,
        "solar_radiation_mj_m2_day_mean": 19.1,
    }

    t_start = time.time()
    payload = pipeline.process_pipeline(sample_features)
    latency = (time.time() - t_start) * 1000

    print(f"   ↳ Prediction Completed in {latency:.2f} ms")

    # 3. Assert Backend JSON Schema Compliance
    print("\n[STEP 3] Validating Web Backend JSON Structure & Modules...")
    required_keys = [
        "status",
        "location_metadata",
        "crop_health_map_layer",
        "crop_recommendation",
        "location_detail_panel",
        "risk_alerts",
        "user_pattern_analysis",
        "market_and_infrastructure",
        "agricultural_gdp_forecast",
        "crop_suitabilities",
        "pipeline_metadata"
    ]

    for key in required_keys:
        assert key in payload, f"Missing required BE key: '{key}'"
        print(f"   ✓ Module '{key}' verified.")

    # 4. Verify JSON Serialization
    print("\n[STEP 4] Testing JSON Serialization...")
    try:
        json_str = json.dumps(payload, indent=2)
        print(f"   ✓ JSON Serialization Successful ({len(json_str):,} characters).")
    except Exception as e:
        print(f"   ✗ JSON Serialization Failed: {e}")
        sys.exit(1)

    # 5. Save Sample JSON Output
    sample_json_path = PROJECT_ROOT / "pipeline" / "sample_output.json"
    with sample_json_path.open("w") as f:
        f.write(json_str)
    print(f"   ↳ Sample BE JSON response saved to: {sample_json_path}")

    # 6. Test with Real Dataset Row if Available
    data_csv = PROJECT_ROOT / "data" / "combined" / "combined_dataset.csv"
    if data_csv.exists():
        print("\n[STEP 5] Testing Pipeline on Real Dataset Row (combined_dataset.csv)...")
        df_row = pd.read_csv(data_csv, nrows=1).iloc[0]
        real_payload = pipeline.process_pipeline(df_row)
        assert real_payload["status"] == "success"
        print(f"   ✓ Real dataset row evaluation successful (Latency: {real_payload['pipeline_metadata']['execution_latency_ms']} ms).")

    # Display Highlights
    print("\n==========================================================================")
    print("                       SAMPLE PIPELINE RESULTS                            ")
    print("==========================================================================")
    print(f" Crop Health Status    : {payload['crop_health_map_layer']['health_status']} ({payload['crop_health_map_layer']['health_score']*100:.1f}%)")
    print(f" Top Crop Recommended  : {payload['crop_recommendation']['top_recommended_crops'][0]['crop_display_name']} (Score: {payload['crop_recommendation']['top_recommended_crops'][0]['rank_score']})")
    print(f" Active Risk Alerts    : {payload['risk_alerts']['alert_count']} alert(s)")
    for a in payload['risk_alerts']['alerts']:
        print(f"   • [{a['severity']}] {a['title']}")
    print(f" Ag GDP Forecast       : ${payload['agricultural_gdp_forecast']['estimated_agricultural_gdp_usd_ha']:,.2f}/ha/year")
    print("==========================================================================\n")
    print("[SUCCESS] All pipeline tests passed cleanly!")


if __name__ == "__main__":
    main()
