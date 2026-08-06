# Myanmar Agricultural Model Serving Microservice

FastAPI inference microservice for the Myanmar Agriculture Intelligence project.
Serves a high-throughput, vectorized dataset processing pipeline (CSV Ingestion → Vectorized Preprocessing → 40 ML Models + 5 Composite Feature Groups → Ultra-Compact Matrix / Parquet Response).

```text
CSV Upload  ──►  Vectorized Ingestion  ──►  40 ML Models + 5 Composites  ──►  Columnar Matrix / Parquet Download
```

---

## ⚡ Performance & Size Optimizations

1. **Async Background Processing (Default)**: Ingests datasets of any size (e.g. 36,000+ rows) asynchronously, returning a `job_id` immediately to eliminate HTTP gateway timeouts.
2. **Columnar Matrix Output (`format=columnar`, Default)**: Eliminates dictionary key repetition per row. Reduces a 36,000-row prediction payload from **230 MB down to ~24 MB** (**90% size reduction**).
3. **Transparent GZip HTTP Compression**: Compresses HTTP responses over 1 KB, reducing 24 MB JSON down to **~2 MB over the wire**.
4. **Parquet Binary Downloads (`GET /pipeline/status/{job_id}/download`)**: Generates a **3 MB Parquet binary export file** upon job completion, allowing full-dataset downloads in **<0.2 seconds**.

---

## 🚀 API Endpoint Reference

### 1. Asynchronous Dataset Ingestion Pipeline (`POST /api/v1/pipeline/run-async`)
Submits a regional CSV dataset for background vectorized inference.

#### Request:
```bash
curl -X POST "http://localhost:8001/api/v1/pipeline/run-async?format=columnar" \
  -F "file=@data/raw/yangon/yangon.csv"
```

#### Response:
```json
{
  "status": "processing",
  "job_id": "job_92341b11e1d2"
}
```

---

### 2. Job Status & Columnar Prediction Fetch (`GET /api/v1/pipeline/status/{job_id}`)
Checks job progress and retrieves completed predictions in ultra-compact matrix format.

#### Request:
```bash
curl "http://localhost:8001/api/v1/pipeline/status/job_92341b11e1d2"
```

#### Response Structure:
```json
{
  "status": "completed",
  "job_id": "job_92341b11e1d2",
  "progress_pct": 100.0,
  "download_parquet_url": "/api/v1/pipeline/status/job_92341b11e1d2/download",
  "result": {
    "status": "success",
    "total_rows": 36672,
    "format": "columnar",
    "meta": {
      "indices": ["1847,432", "1847,433"],
      "lats": [17.2016, 17.2426],
      "lons": [95.7390, 95.7800],
      "regions": ["yangon", "yangon"]
    },
    "predictions": {
      "crop_suitability_monsoon_rice": ["good", "good"],
      "crop_health_score": [0.65, 0.72]
    },
    "composite_features": {
      "crop_recommender": [[ { "crop": "monsoon_rice", "suitability": "good" } ]],
      "crop_health": [ { "health_score": 0.65, "health_class": "Good" } ]
    },
    "pipeline_metadata": {
      "execution_time_ms": 122670.34,
      "total_predictions_evaluated": 1466880,
      "models_used_count": 1466880,
      "fallbacks_used_count": 0
    }
  }
}
```

---

### 3. Streamed Parquet Binary File Download (`GET /api/v1/pipeline/status/{job_id}/download`)
Downloads predictions as an ultra-compact **~3 MB Parquet binary file**.

#### Request:
```bash
curl -O -J "http://localhost:8001/api/v1/pipeline/status/job_92341b11e1d2/download"
```

---

### 4. Supporting Health & Diagnostics Endpoints
- `GET /api/v1/live`: Liveness probe (`{"status": "alive"}`)
- `GET /api/v1/ready`: Readiness probe (`{"status": "ready", "model_targets_count": 40}`)
- `GET /api/v1/health`: Resource & memory diagnostics (`{"ram_usage_mb": 2180.6, "loaded_models_in_ram": 40}`)
- `GET /api/v1/models`: List of all 40 prediction target definitions

---

## 💻 CLI Client Helper Usage

Test dataset uploads using the python CLI client:
```bash
# Default Async run with Columnar matrix output:
python scripts/test_pipeline.py --csv data/raw/yangon/yangon.csv --output result.json

# Process full regional dataset and download 3MB Parquet binary directly:
python scripts/test_pipeline.py --csv data/raw/yangon/yangon.csv --output result.json --limit -1 --parquet
```

---

## 🐳 Docker Deployment

Exposes the model API microservice on host port `8001`.

```bash
# Create writeable host jobs directory
mkdir -p jobs && chmod 777 jobs

# Start container
docker compose up --build -d
```
