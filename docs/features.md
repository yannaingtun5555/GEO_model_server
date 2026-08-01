#	Feature Name	Description	Units	Why It Matters
1	elevation_m	Height above sea level	meters	Determines temperature, water drainage, crop suitability
2	slope_degrees	Steepness of terrain	degrees	Affects erosion risk, water runoff, machinery access
3	aspect_degrees	Direction slope faces (0-360°)	degrees	Determines sun exposure, affects temperature and moisture
4	distance_to_surface_water_m	Distance to nearest water body	meters	Irrigation access, flood risk
5	soil_cec_cmol_kg_0_30cm	Cation Exchange Capacity (nutrient holding ability)	cmol/kg	Higher = better nutrient retention
6	soil_clay_pct_0_30cm	Clay content in top 30cm	percentage	Affects water retention, nutrient holding
7	soil_sand_pct_0_30cm	Sand content in top 30cm	percentage	Affects drainage, erosion risk
8	soil_silt_pct_0_30cm	Silt content in top 30cm	percentage	Affects soil texture, water holding
9	soil_soc_g_kg_0_30cm	Soil Organic Carbon	g/kg	Key indicator of soil fertility
10	soil_ph_h2o_0_30cm	Soil pH (acidity/alkalinity)	pH (0-14)	Affects nutrient availability
11	surface_water_occurrence_pct	How often water is present	percentage	Indicates flood risk, water availability
12	surface_water_seasonality_months	Months when water is present	months (1-12)	Water availability pattern
13	latitude	North-South coordinate	degrees	Climate zone, growing season length
14	longitude	East-West coordinate	degrees	Regional climate patterns

🌧️ DYNAMIC AGGREGATED FEATURES (27)

These are calculated from 12 monthly values. For each location, you need data for all 12 months.
A. Precipitation (5 features)

Base data: Monthly rainfall in mm from CHIRPS
#	Feature	Calculation	What It Tells You
15	chirps_precipitation_mm_mean	Average of 12 months	Total annual rainfall / 12 - how much rain normally falls
16	chirps_precipitation_mm_max	Maximum monthly value	Peak rainfall month - helps identify flood risk
17	chirps_precipitation_mm_min	Minimum monthly value	Driest month - helps identify drought risk
18	chirps_precipitation_mm_range	Max - Min	How much rainfall varies through the year
19	chirps_precipitation_mm_cv	Std Dev / Mean	Coefficient of variation - how predictable/unpredictable rainfall is

B. Soil Moisture (4 features)

Base data: Monthly soil moisture from ERA5-Land (m³/m³)
#	Feature	Calculation	What It Tells You
20	era5_soil_moisture_m3_m3_mean	Average of 12 months	Normal soil moisture level
21	era5_soil_moisture_m3_m3_max	Maximum monthly value	Wettest conditions
22	era5_soil_moisture_m3_m3_min	Minimum monthly value	Driest conditions
23	era5_soil_moisture_m3_m3_cv	Std Dev / Mean	How much moisture varies through the year


C. Temperature (4 features)

Base data: Monthly mean temperature from ERA5-Land (°C)
#	Feature	Calculation	What It Tells You
24	mean_temperature_c_mean	Average of 12 months	Normal growing temperature
25	mean_temperature_c_max	Maximum monthly value	Hottest month
26	mean_temperature_c_min	Minimum monthly value	Coolest month
27	mean_temperature_c_range	Max - Min	Temperature seasonality

D. NDVI - Vegetation Health (4 features)

Base data: Monthly NDVI (Normalized Difference Vegetation Index) from Sentinel-2
#	Feature	Calculation	What It Tells You
28	ndvi_median_mean	Average of 12 months	Overall vegetation health
29	ndvi_median_max	Maximum monthly value	Peak vegetation (crop growth period)
30	ndvi_median_min	Minimum monthly value	When vegetation is at its lowest
31	ndvi_median_growing_season_mean	Average of months where NDVI > 0.4	Vegetation health during growing season

E. NDWI - Water Index (2 features)

Base data: Monthly NDWI (Normalized Difference Water Index) from Sentinel-2
#	Feature	Calculation	What It Tells You
32	ndwi_mcf_median_mean	Average of 12 months	Overall water availability
33	ndwi_mcf_median_max	Maximum monthly value	Wettest period

F. Radar Backscatter - Surface Properties (2 features)

Base data: Monthly Sentinel-1 VH and VV backscatter (dB)
#	Feature	Calculation	What It Tells You
34	s1_vh_db_median_mean	Average of 12 months	Surface roughness, crop structure
35	s1_vv_db_median_mean	Average of 12 months	Surface moisture, soil properties

G. Solar Radiation (2 features)

Base data: Monthly solar radiation from ERA5-Land (MJ/m²/day)
#	Feature	Calculation	What It Tells You
36	solar_radiation_mj_m2_day_mean	Average of 12 months	Overall solar energy for photosynthesis
37	solar_radiation_mj_m2_day_max	Maximum monthly value	Peak solar energy period

