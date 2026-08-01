┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Web App)                     │
│  Farmer inputs: Location (lat/lon) or clicks on map             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 DATA RETRIEVAL LAYER                            │
│  - Fetch static features for location                           │
│  - Fetch dynamic features for ALL months (Jan-Dec 2018)         │
│  - Create time-series feature vector                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              CORE ML MODEL (Multi-Output)                       │
│  Input: 28 static features + (12 months × 16 dynamic)           │
│  Output: 4 predictions                                          │
│  ├─ Crop Type (Classification: 4 classes)                       │
│  ├─ Health Score (Regression: 0-1)                              │
│  ├─ Yield (Regression: tons/ha)                                 │
│  └─ Irrigation Need (Classification: Low/Medium/High)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION FORMATTER                         │
│  Convert raw predictions to structured data:                    │
│  {crop: "Rice", health: 0.72, yield: 4.2, irrigation: "M"}      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              GEMMA 4 ADVISORY GENERATOR                         │
│  - Takes structured predictions + location data                 │
│  - Generates farmer-friendly advisory in Burmese/English        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              FINAL OUTPUT (Display to Farmer)                   │
│  "Based on your location in Ayeyawaddy...                       │
│   We recommend planting Rice this season.                       │
│   Expected yield: 4.2 tons/ha                                   │
│   Current crop health: Good (0.72/1.0)                          │
│   Irrigation need: Medium - schedule watering every 5 days      │
│   ... [detailed advice]"                                        │
└─────────────────────────────────────────────────────────────────┘
