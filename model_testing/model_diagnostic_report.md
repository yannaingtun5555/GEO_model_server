# Machine Learning Model Diagnostic & Health Report

## Executive Summary
- **Total Models Evaluated**: 40
- **Healthy Models (Passed All Diagnostic Checks)**: **31** (77.5%)
- **Flagged Models (Requires Review/Refinement)**: **9** (22.5%)

## Overall Model Health Breakdown

| Target Model | Type | Overall Health | Issues Count | Train Score | Test Score | Generalization Gap | Issues Summary |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `agricultural_gdp_forecast` | Regression | 🟢 HEALTHY | 0 | 0.9935 | 0.9936 | -0.0001 | None (Optimal Model Health) |
| `agricultural_land_conversion_risk` | Regression | 🟡 FLAGGED | 1 | 0.9961 | 0.9962 | -0.0001 | HIGH OUTLIER ERRORS: 137 test predictions (2.28%) have errors > 3x RMSE; HETEROSCEDASTICITY DETECTED: Residual magnitude correlates with predicted scale (r=0.5430) |
| `cold_chain_potential` | Regression | 🟡 FLAGGED | 1 | 0.9962 | 0.9963 | -0.0001 | HIGH OUTLIER ERRORS: 111 test predictions (1.85%) have errors > 3x RMSE |
| `crop_health_score` | Regression | 🟢 HEALTHY | 0 | 0.9886 | 0.9888 | -0.0002 | None (Optimal Model Health) |
| `crop_suitability_black_gram` | Classification | 🟢 HEALTHY | 0 | 0.9926 | 0.9928 | -0.0002 | None (Optimal Model Health) |
| `crop_suitability_cassava` | Classification | 🟢 HEALTHY | 0 | 0.9921 | 0.9932 | -0.0010 | None (Optimal Model Health) |
| `crop_suitability_chili` | Classification | 🟢 HEALTHY | 0 | 0.9948 | 0.9959 | -0.0011 | None (Optimal Model Health) |
| `crop_suitability_dry_season_rice` | Classification | 🟢 HEALTHY | 0 | 0.9786 | 0.9799 | -0.0013 | None (Optimal Model Health) |
| `crop_suitability_durian` | Classification | 🟢 HEALTHY | 0 | 0.9772 | 0.9788 | -0.0016 | None (Optimal Model Health) |
| `crop_suitability_green_gram` | Classification | 🟢 HEALTHY | 0 | 0.9908 | 0.9913 | -0.0005 | None (Optimal Model Health) |
| `crop_suitability_groundnut` | Classification | 🟢 HEALTHY | 0 | 0.9911 | 0.9904 | 0.0008 | None (Optimal Model Health) |
| `crop_suitability_longan` | Classification | 🟢 HEALTHY | 0 | 0.9825 | 0.9825 | -0.0000 | None (Optimal Model Health) |
| `crop_suitability_maize` | Classification | 🟢 HEALTHY | 0 | 0.9904 | 0.9922 | -0.0018 | None (Optimal Model Health) |
| `crop_suitability_mango` | Classification | 🟢 HEALTHY | 0 | 0.9914 | 0.9900 | 0.0014 | None (Optimal Model Health) |
| `crop_suitability_mangosteen` | Classification | 🟢 HEALTHY | 0 | 0.9744 | 0.9753 | -0.0009 | None (Optimal Model Health) |
| `crop_suitability_monsoon_rice` | Classification | 🟢 HEALTHY | 0 | 0.9854 | 0.9835 | 0.0019 | None (Optimal Model Health) |
| `crop_suitability_pigeon_pea` | Classification | 🟢 HEALTHY | 0 | 0.9897 | 0.9888 | 0.0009 | None (Optimal Model Health) |
| `crop_suitability_rubber` | Classification | 🟢 HEALTHY | 0 | 0.9938 | 0.9941 | -0.0003 | None (Optimal Model Health) |
| `crop_suitability_sesame` | Classification | 🟢 HEALTHY | 0 | 0.9881 | 0.9875 | 0.0006 | None (Optimal Model Health) |
| `crop_suitability_sugarcane` | Classification | 🟢 HEALTHY | 0 | 0.9900 | 0.9901 | -0.0001 | None (Optimal Model Health) |
| `crop_suitability_tomato` | Classification | 🟢 HEALTHY | 0 | 0.9392 | 0.9420 | -0.0028 | None (Optimal Model Health) |
| `crop_yield_t_ha` | Regression | 🟢 HEALTHY | 0 | 0.9839 | 0.9836 | 0.0002 | None (Optimal Model Health) |
| `current_month_mean_temperature_c` | Regression | 🟡 FLAGGED | 1 | 0.9801 | 0.9800 | 0.0001 | HIGH OUTLIER ERRORS: 87 test predictions (1.45%) have errors > 3x RMSE |
| `current_month_precipitation_mm` | Regression | 🟡 FLAGGED | 2 | 0.9530 | 0.9518 | 0.0012 | TARGET LEAKAGE SUSPECTED: Features near-identically correlated with target: chirps_precipitation_mm (r=0.9995) | HIGH OUTLIER ERRORS: 130 test predictions (2.17%) have errors > 3x RMSE; HETEROSCEDASTICITY DETECTED: Residual magnitude correlates with predicted scale (r=0.5604) |
| `current_month_solar_rad_mj_m2_day` | Regression | 🟡 FLAGGED | 1 | 0.9677 | 0.9679 | -0.0001 | HIGH OUTLIER ERRORS: 82 test predictions (1.37%) have errors > 3x RMSE |
| `drought_risk_score` | Regression | 🟢 HEALTHY | 0 | 0.9965 | 0.9965 | -0.0000 | None (Optimal Model Health) |
| `flood_risk_level` | Classification | 🟢 HEALTHY | 0 | 1.0000 | 1.0000 | 0.0000 | None (Optimal Model Health) |
| `heat_stress_risk` | Classification | 🟢 HEALTHY | 0 | 1.0000 | 1.0000 | 0.0000 | None (Optimal Model Health) |
| `irrigation_need` | Classification | 🟢 HEALTHY | 0 | 0.9978 | 0.9982 | -0.0004 | None (Optimal Model Health) |
| `irrigation_potential` | Regression | 🟢 HEALTHY | 0 | 0.9770 | 0.9773 | -0.0002 | None (Optimal Model Health) |
| `market_integration_score` | Regression | 🟢 HEALTHY | 0 | 0.9925 | 0.9929 | -0.0004 | None (Optimal Model Health) |
| `nitrogen_requirement_level` | Classification | 🟢 HEALTHY | 0 | 1.0000 | 1.0000 | 0.0000 | None (Optimal Model Health) |
| `optimal_planting_month` | Classification | 🟢 HEALTHY | 0 | 1.0000 | 1.0000 | 0.0000 | None (Optimal Model Health) |
| `phosphorus_requirement_level` | Classification | 🟢 HEALTHY | 0 | 1.0000 | 1.0000 | 0.0000 | None (Optimal Model Health) |
| `post_harvest_loss_risk` | Regression | 🟢 HEALTHY | 0 | 0.9778 | 0.9782 | -0.0004 | None (Optimal Model Health) |
| `soil_erosion_risk` | Classification | 🟢 HEALTHY | 0 | 1.0000 | 1.0000 | 0.0000 | None (Optimal Model Health) |
| `supply_chain_efficiency` | Regression | 🟡 FLAGGED | 1 | 0.9953 | 0.9956 | -0.0003 | HIGH OUTLIER ERRORS: 62 test predictions (1.03%) have errors > 3x RMSE |
| `surface_water_occurrence` | Regression | 🟡 FLAGGED | 1 | 0.9805 | 0.9810 | -0.0005 | HIGH OUTLIER ERRORS: 77 test predictions (1.28%) have errors > 3x RMSE |
| `urban_encroachment_risk` | Regression | 🟡 FLAGGED | 1 | 0.9937 | 0.9942 | -0.0004 | HIGH OUTLIER ERRORS: 62 test predictions (1.03%) have errors > 3x RMSE |
| `water_scarcity_risk` | Regression | 🟡 FLAGGED | 1 | 0.9935 | 0.9936 | -0.0001 | HIGH OUTLIER ERRORS: 99 test predictions (1.65%) have errors > 3x RMSE |


## Diagnostic Verification Methodology
1. **Overfitting Check**: Compares Train vs. Test score gap ($> 5\%$ delta flags overfitting).
2. **Data Leakage Check**: Identifies target correlation ($|r| > 0.999$) and sample overlap.
3. **Class Imbalance Check**: Flags 0% precision/recall classes and baseline lift.
4. **Physical Boundaries Check**: Tests for negative yields or out-of-range physical values.
5. **Residual Bias Check**: Tests mean bias and error heteroscedasticity.
