# Backend Engineer's Guide — Model Serving Microservice & Ingestion Architecture

This document provides technical instructions for backend engineers managing the model serving microservice, dataset ingestion, and prediction pipeline exports.

---

## 1. Pipeline Ingestion & Processing Flow

Satellite observation CSV exports (containing grid coordinates and dynamic/static observations) are ingested via the model server endpoints:

- **Asynchronous Endpoint (Primary/Default)**: `POST /api/v1/pipeline/run-async`
- **Synchronous Endpoint**: `POST /api/v1/pipeline/run`

### Workflow:
```text
[ GEE CSV Uploaded ]
        │
        ▼ (FastAPI Background Task)
[ Vectorized Data Alignment & Type Casting ]
        │
        ▼
[ Vectorized Batch Predictions (All 40 Models) ]
        │
        ▼
[ Composite Feature Calculations (5 Groups) ]
        │
        ▼
[ Save /app/jobs/{job_id}.json (Columnar) & /app/jobs/{job_id}.parquet ]
```

---

## 2. API Endpoints for Integration

### A. Submitting Dataset for Background Execution
`POST /api/v1/pipeline/run-async`
- **Header**: `Content-Type: multipart/form-data`
- **Query Parameter**: `format=columnar` (default matrix format for ~90% size reduction)
- **Response**: `{"status": "processing", "job_id": "job_1a36d41b5849"}`

### B. Polling Job Progress & Fetching JSON Predictions
`GET /api/v1/pipeline/status/{job_id}`
- **Response status**: `processing` (progress_pct: 10.0) → `completed` (progress_pct: 100.0)
- Contains the full Columnar matrix predictions envelope and a direct URL to the Parquet binary download.

### C. Direct Parquet Binary File Download
`GET /api/v1/pipeline/status/{job_id}/download`
- **Content-Type**: `application/x-parquet`
- **Performance**: Streams a **~3 MB Parquet binary file** containing 33,000+ rows directly into memory in **<0.2 seconds**.

```python
import pandas as pd

# Load 33,000 prediction rows directly into a Pandas DataFrame:
df = pd.read_parquet("http://localhost:8001/api/v1/pipeline/status/job_1a36d41b5849/download")
```

---

## 3. High-Throughput & Resource Controls

### RAM Management (`MAX_RAM_MB=2400`)
- Model artifacts are loaded into an LRU cache.
- Python memory usage is budgeted at `MAX_RAM_MB=2400` to prevent container OOM termination under Docker's `mem_limit: 3g`.

### Output Storage (`/app/jobs`)
- Background output payloads are saved to `/app/jobs` (mounted to host `./jobs`).
- Prevents container memory bloat and protects container file system limits.

### Automatic HTTP Gzip Compression
- All JSON payloads over 1 KB are transparently Gzip compressed by FastAPI middleware.
