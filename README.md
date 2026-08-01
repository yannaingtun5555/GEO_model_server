# Myanmar Agricultural Model Serving Microservice & ML Pipeline

A high-performance, memory-efficient Standalone Model Serving Microservice and Machine Learning Pipeline built for agricultural intelligence, crop suitability ranking, multi-hazard risk assessment, and yield forecasting across Myanmar.

---

## 📊 Datasets Used for Predictions

The pipeline uses a structured dataset hierarchy optimized for high-speed spatial lookup and model inference:

| Dataset | Location | File Size | Description & Usage |
| :--- | :--- | :--- | :--- |
| **Inference Feature Dataset** *(Primary for Serving)* | `data/processed/features_dataset.parquet` | ~52 MB | **Used directly by the Model Server for predictions.** Contains lightweight input feature vectors (satellite observation indicators, climate metrics, topography, soil properties) stripped of target labels. Indexed with a spatial KD-Tree for $O(\log N)$ nearest-neighbor coordinate lookup. |
| **Full Combined Dataset** | `data/combined/combined_dataset.csv` | ~1.21 GB | Full merged dataset containing input features and engineered target labels across all 6 target regions from 2018–2026. Used for training ML models (`train.py`) and full accuracy evaluation. *(Git Ignored)* |
| **Regional Processed CSVs** | `data/processed/{region}/{year}/*.csv` | Multi-GB | Intermediate monthly feature/label CSV files processed per region (`ayeyawaddy`, `bago`, `magway`, `mandalay`, `sagaing`, `yangon`). *(Git Ignored)* |
| **Raw Earth Engine CSVs** | `data/raw/{region}/{year}/*.csv` | Multi-GB | Raw Google Earth Engine observation CSVs (Landsat, Sentinel, CHIRPS, ERA5, MODIS, Copernicus DEM). *(Git Ignored)* |

> 💡 **Git Ignore Note:** Large `.csv` files and `data/raw/` directories are ignored in `.gitignore` to keep the repository lightweight for pushing to remote Git servers (e.g., GitHub 100MB limit).

---

## 🛠️ Model Server Technologies & Architecture

The model serving backend is engineered for ultra-low latency inference, constrained memory footprint, and high throughput.

```
                  ┌─────────────────────────────────────────┐
                  │          FastAPI Microservice           │
                  │             (server/main.py)            │
                  └────────────────────┬────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│ Spatial KD-   │              │   LRU Model   │              │  Redis Cache  │
│ Tree Engine   │              │    Manager    │              │    Service    │
│  (cKDTree)    │              │ (2GB RAM Cap) │              │  (24h TTL)    │
└───────┬───────┘              └───────┬───────┘              └───────┬───────┘
        │                              │                              │
        ▼                              ▼                              ▼
Feature Lookup                40 Trained Models /             Cached Response
(Lat/Lon → Vector)            Fallback Estimators              (Sub-ms Return)
```

### Key Technologies:
- **[FastAPI](https://fastapi.tiangolo.com/) & Uvicorn**: Asynchronous Python web framework providing high-concurrency REST API endpoints.
- **Spatial KD-Tree Indexing (`scipy.spatial.cKDTree`)**: $O(\log N)$ spatial coordinate (`latitude`, `longitude`) nearest-neighbor lookup engine to retrieve environmental & satellite feature vectors instantly.
- **LRU Memory-Capped Model Loader**: Dynamic Least Recently Used (LRU) model manager capping active model memory usage to **max 4 heavy models / 2048 MB RAM** to prevent Out-Of-Memory (OOM) failures in production containers.
- **Redis Caching (`redis-py`)**: Ultra-fast key-value cache layer with configurable TTL (24 hours default) for repeated coordinate or regional query payloads.
- **Scikit-Learn & Joblib**: Machine learning execution engine supporting Random Forest & Gradient Boosting models across 40 agricultural targets.
- **Composite Intelligence Engine**: Rule-based & composite scoring engine that transforms model outputs into actionable insights:
  - **Crop Recommender**: Ranks 17 crops based on suitability scores & economic yields.
  - **Multi-Hazard Alert**: Aggregates flood, drought, heat stress, and soil erosion risks.
  - **Economic ROI Calculator**: Estimates expected return on investment.
  - **Land Use Change Monitor**: Analyzes agricultural conversion & urban encroachment risks.
- **Docker & Docker Compose**: Multi-stage containerization setup with automated health checks and Redis service orchestration.

---

## 🚀 Getting Started & Setup

### Prerequisites
- Python 3.10+
- Redis (optional, fallback to in-memory dictionary cache if unavailable)
- Docker & Docker Compose (optional for containerized deployment)

### 1. Local Setup

Clone the repository and install dependencies:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/python3

# Install requirements
pip install -r requirements.txt
```

### 2. Export Feature Dataset for Inference (Optional)

If `data/processed/features_dataset.parquet` is missing or you modified `data/combined/combined_dataset.csv`, export the lightweight feature dataset:

```bash
python scripts/export_inference_features.py
```

### 3. Run the Model Server

#### Option A: Direct Python / Uvicorn

```bash
python -m server.main
# Or: uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Option B: Using Convenience Script (`run.sh`)

```bash
./run.sh pipeline
```

#### Option C: Docker Container Deployment

```bash
# Build and run Model Server + Redis containers
docker compose up --build -d
```

The server will start at: `http://localhost:8000`
API Documentation (Swagger UI): `http://localhost:8000/docs`

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/predict` | Main inference endpoint. Accepts `lat`/`lon`, `system_index`, or `region_name`. Returns predictions & composite features. |
| `GET` | `/api/v1/regions/{region_name}` | Returns pre-computed regional crop suitability ranking and climate summary. |
| `GET` | `/api/v1/health` | Health diagnostic check showing current RAM usage, loaded LRU models, and Redis status. |

### Example Request (`POST /api/v1/predict`)

```json
{
  "lat": 16.8661,
  "lon": 96.1951,
  "include_all_targets": true,
  "composite_features": ["crop_recommender", "risk_alerts", "economic_roi"]
}
```

### Example Response Payload

```json
{
  "status": "success",
  "location": {
    "system_index": "00000000000000000001",
    "lat": 16.8661,
    "lon": 96.1951,
    "nearest_distance_deg": 0.00234
  },
  "predictions": {
    "crop_yield_t_ha": 4.12,
    "crop_health_score": 82.5,
    "drought_risk_score": 0.15,
    "crop_suitability_monsoon_rice": "excellent",
    "crop_suitability_maize": "good"
  },
  "composite_features": {
    "crop_recommender": {
      "top_recommendations": [
        {"crop": "monsoon_rice", "suitability": "excellent", "suitability_score": 1.0},
        {"crop": "maize", "suitability": "good", "suitability_score": 0.75}
      ]
    }
  },
  "execution_metadata": {
    "response_time_ms": 14.5,
    "cached": false,
    "ram_used_mb": 245.2,
    "lru_models_in_memory": ["crop_yield_t_ha", "crop_health_score"]
  }
}
```

---

## 🎯 40 ML Models Overview

The pipeline supports predictions across 40 target variables:
1. **17 Crop Suitabilities**: Monsoon Rice, Dry Season Rice, Maize, Sugarcane, Cassava, Durian, Mangosteen, Longan, Mango, Chili, Tomato, Black Gram, Green Gram, Pigeon Pea, Groundnut, Sesame, Rubber.
2. **Core Agronomic Indicators**: Crop Health Score, Crop Yield (t/ha), Irrigation Need.
3. **Climate Forecasts**: Monthly Precipitation, Mean Temperature, Solar Radiation.
4. **Environmental Hazards**: Flood Risk Level, Drought Risk Score, Heat Stress Risk, Soil Erosion Risk.
5. **Management & Soil**: Optimal Planting Month, Nitrogen Requirement, Phosphorus Requirement.
6. **Market & Infrastructure**: Market Integration Score, Post-Harvest Loss Risk, Supply Chain Efficiency, Cold Chain Potential.
7. **Land Use & Water**: Agricultural Land Conversion Risk, Urban Encroachment Risk, Irrigation Potential, Surface Water Occurrence, Water Scarcity Risk.
8. **Economic Output**: Agricultural GDP Forecast.

---

## 📄 License

Internal / Production Proprietary — Myanmar Agricultural Intelligence Project.
