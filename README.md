# Myanmar Agricultural Experimental Model Server

Private FastAPI inference microservice for the Myanmar Agriculture Intelligence project.
Serves a streamlined dataset processing pipeline (CSV → Preprocess → Predict 40 Models + Composites → Response) for web backend integration.

```text
Web App / Daily Task -> CSV Upload -> Preprocess -> Predict 40 ML Models + Composites -> Direct JSON Response
```

## 🛰️ Dataset Pipeline API (`POST /api/v1/pipeline/run`)

The primary pipeline API accepts a daily uploaded CSV dataset (matching `data/test/*.csv` or `data/raw/yangon/yangon.csv`), processes it, evaluates **all 40 machine learning models**, computes **all 5 composite feature groups**, and directly returns per-index structured predictions.

### Workflow:
1. **CSV Upload**: Accepts daily region export CSV via multipart file upload.
2. **Preprocessing**: Automatically extracts index IDs (`system:index`/`sample_id`), `latitude`, `longitude`, and `region`, aligns 75 feature columns, and handles missing input imputation.
3. **40 Model Predictions**: Runs model inference for all 40 ML targets across every row.
4. **5 Composite Feature Groups**: Computes composite features per row:
   - `crop_recommender`: Suitability rankings & scores across all 17 crops.
   - `crop_health`: Health layer score, classification, and NDVI analysis.
   - `economic_roi`: Economic ROI calculation indicator.
   - `risk_alerts`: Multi-hazard alerts (Flood, Drought, Heat, Erosion, Water Scarcity).
   - `land_use`: Land conversion and urban encroachment risk analysis.
5. **Direct Response**: Returns structured predictions and metadata for every land index.

---

## 🚀 API Endpoints

### 1. Main Pipeline API
- `POST /api/v1/pipeline/run`: Ingest CSV dataset, preprocess, evaluate all 40 models + 5 composite groups per row, and return predictions.

#### Example Request:
```bash
curl -F "file=@data/raw/yangon/yangon.csv" http://localhost:8001/api/v1/pipeline/run
```

#### Example Response Structure:
```json
{
  "status": "success",
  "total_rows": 2,
  "rows": [
    {
      "meta": {
        "index": "1847,432",
        "sample_id": "mm_1847_432__2018-01",
        "lat": 17.20167,
        "lon": 95.73900,
        "region": "yangon"
      },
      "predictions": {
        "crop_health_score": { "value": 0.54, "task_type": "regression", "label": "0.54", "is_fallback": false },
        "crop_suitability_monsoon_rice": { "value": "good", "task_type": "classification", "label": "good", "is_fallback": false },
        "...": "38 more targets..."
      },
      "composite_features": {
        "crop_recommender": [ { "crop": "monsoon_rice", "suitability": "good", "suitability_score": 75.0 } ],
        "crop_health": { "health_score": 0.54, "health_class": "Good", "map_color_hex": "#3B82F6" },
        "economic_roi": { "status": "unavailable" },
        "risk_alerts": { "overall_level": "low", "risk_scores": { ... } },
        "land_use": { "risk_level": "low", "conversion_risk_score": 0.0 }
      }
    }
  ],
  "pipeline_metadata": {
    "filename": "yangon.csv",
    "execution_time_ms": 145.2,
    "total_predictions_evaluated": 80,
    "models_used_count": 80,
    "fallbacks_used_count": 0
  }
}
```

### 2. Supporting Microservice APIs
- `GET /api/v1/live`: Liveness probe (`{"status": "alive"}`)
- `GET /api/v1/ready`: Server readiness check (`{"status": "ready", "model_targets_count": 40}`)
- `GET /api/v1/health`: Resource & memory diagnostics
- `GET /api/v1/models`: Returns authoritative list of all 40 prediction target definitions

---

## 🛠️ Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh serve
```

The model API will be available at `http://127.0.0.1:8001`.

### Run Verification Tests
```bash
.venv/bin/python -c "
from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)
print(client.get('/api/v1/live').json())
print(client.get('/api/v1/ready').json())
"
```

---

## 🐳 Docker Deployment

Exposes the model API microservice on host port `8001`.

```bash
docker compose up --build
```
