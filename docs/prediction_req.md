prediction 1-17
17 cols for each crops
(moonsun rice,dry-season-rice,maize,sugarcane,cassava,durial,mangosteen,longan,mango,chili,tomato,black gram,green gram,pigean_pea,groundnt,sesame,rubber)
with 4 types for each col(crop) ->excellent,good,modrate,poor

prediction 18: Crop Health Score

What it predicts: How healthy the crop will be at that location (0-1 scale).
Score Range	Health Status	What It Means
0.80 - 1.00	Excellent	Optimal conditions, minimal stress, high productivity
0.60 - 0.79	Good	Generally favorable, minor manageable issues
0.40 - 0.59	Moderate	Some stress factors present, needs attention
0.20 - 0.39	Poor	Significant stress, high risk of yield loss
0.00 - 0.19	Critical	Severe stress, likely crop failure

Factors Considered:

    NDVI (vegetation vigor)

    Soil moisture availability

    Temperature stress

    Water stress (NDWI)

    Soil fertility

Output Type: Regression (0.0 - 1.0)
Example: health_score: 0.72 → "Good"

prediction 19: Crop Yield Prediction

What it predicts: Expected harvest amount in tons per hectare (tons/ha).
Yield Range	Interpretation
5.0 - 6.0	Excellent yield (highly productive)
4.0 - 4.9	Good yield (above average)
3.0 - 3.9	Average yield (typical for region)
2.0 - 2.9	Below average (some constraints)
0.5 - 1.9	Poor yield (significant limitations)

Factors Considered:

    Soil fertility (SOC, CEC, pH)

    Precipitation (amount and distribution)

    Temperature

    Vegetation health (NDVI)

    Soil moisture

Typical Values for Ayeyawaddy:

    Rice: 3.5 - 5.5 tons/ha

    Maize: 2.5 - 4.5 tons/ha

    Pulses: 1.5 - 3.0 tons/ha

    Oilseeds: 1.0 - 2.5 tons/ha

Output Type: Regression (tons/ha)
Example: yield_prediction: 4.2 → 4.2 tons/ha

prediction 20: Irrigation Need Classification 

What it predicts: How urgently the crop needs irrigation.
Code	Category	Meaning	Action Required
0	Low	Sufficient natural water	No irrigation needed, monitor weekly
1	Medium	Moderate water deficit	Supplementary irrigation, water every 5-7 days
2	High	Significant water deficit	Urgent irrigation needed, water every 2-3 days

Factors Considered:

    Precipitation (current and seasonal)

    Soil moisture

    NDWI (water index)

    Soil texture (sand = more need, clay = less)

    Surface water availability

Output Type: Classification (3 classes)
Example: irrigation_need: 1 → "Medium"...


prediction 21-current_month_precipitation
prediction 22-current_month_mean_tempreature
prediction 23-current_month_mean_solar_rad
pred24-flood-risk-level
pred25-drought stress index
pred26-heat stress indexc
pred27-optimal planitng month
pred28-nitrogen_requirement_level
pred29-phosphorus_requirement_level
pred30-soil_erosion_risk
new features

Market Access & Supply Chain Models
Model	Target Variable	Features to Use
31-Market Integration Score	Distance to roads/railways	distance_to_road_km, distance_to_railway_km, road_density, railway_density
32-Post-Harvest Loss Risk	Transportation access	Road density + crop area + distance to rivers
33-Supply Chain Efficiency	Combined transport score	All infrastructure features + population density
34-Cold Chain Potential	Infrastructure readiness	Road density + urban fraction + builtup_fraction

Urbanization & Land Use Change Models
35-Agricultural Land Conversion Risk,
Model	Description	Key Features
36-Urban Encroachment Risk	,Probability of farmland conversion to urban	urban_fraction, builtup_fraction, population_density, road_density

With River/Water Access
Model	Target	Features
37-Irrigation Potential,	Water availability + access	distance_to_river_km, river_density, 38-surface_water_occurrence, permanent_water_fraction
39-Water Scarcity Risk,	Combined water stress	River density + precipitation + soil moisture

40-atricultural gdp forcast model