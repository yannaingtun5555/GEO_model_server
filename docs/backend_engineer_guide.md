# Backend Engineer's Guide — Live GEE Data Pipeline & Prediction Serving

This document provides technical instructions for the backend engineer responsible for maintaining the Google Earth Engine (GEE) update pipeline and managing predictions serving.

---

## 1. GEE Data Export Schedule

As a backend engineer, you will manage GEE data export pipelines. Satellite observation updates (such as Sentinel-1 backscatter, CHIRPS precipitation, MODIS NDVI, ERA5 soil moisture) should be exported on a regular schedule:
- **Frequency**: Once every 1 to 2 weeks.
- **Export Format**: A single combined CSV containing grid point features.
- **Required Identifier Column**: Must contain a `system:index` or `index` column corresponding to the unique identifier of the land grids.
- **Required Dynamic Columns**: Must contain the dynamic observation features (e.g., `chirps_precipitation_mm`, `mean_temperature_c`, `ndvi_median_mean`, `ndwi_mcf_median_mean`, `era5_soil_moisture_m3_m3_mean`, etc.).

---

## 2. API Ingestion Endpoint

Whenever GEE finishes exporting a new dynamic observation dataset, you trigger the update pipeline on the model server by uploading the CSV file.

### Ingestion API Details
- **Route**: `POST /api/v1/pipeline/update`
- **Content-Type**: `multipart/form-data`
- **Payload Parameter**: `file` (the exported CSV file)

### Example `curl` Request:
```bash
curl -X POST "http://localhost:8000/api/v1/pipeline/update" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/gee_dynamic_update_2026_08.csv"
```

### Ingestion Response:
```json
{
  "status": "accepted",
  "message": "GEE update ingestion started in background. Live predictions will be updated on completion.",
  "filename": "gee_dynamic_update_2026_08.csv"
}
```

---

## 3. Background Processing & Lifecycle

Because batch predictions over 1,000,000+ points across all 40 ML models can take 30–60 seconds, the ingestion process is executed asynchronously in the background.

```
[ GEE CSV Uploaded ]
        │
        ▼ (FastAPI Background Task triggers)
[ Merge dynamic GEE columns with master static features (elevation, slope, soils, etc.) ]
        │
        ▼
[ Run Batch predictions on all 40 targets (17 crops + 23 regression/classification) ]
        │
        ▼
[ Save live_predictions.parquet & live_predictions.csv ]
        │
        ▼
[ Regenerate server/static/map_recommendations.json (All 17 crops sorted descending) ]
        │
        ▼
[ Hot-reload live_prediction_manager cache in running memory ]
```

---

## 4. Serving Architecture

To handle high traffic with sub-millisecond response times, the `/predict` API switches dynamically between two modes:

### A. Live Database Mode (Default after Ingestion)
If `live_predictions.parquet` is successfully loaded into memory:
1. When `/predict` receives a request by `system_index` or `(lat, lon)`, it performs an $O(1)$ dictionary lookup or an $O(\log N)$ KD-Tree coordinate query.
2. The server constructs the response directly from the database and returns it in **<5ms**.
3. Response metadata includes `"live_db_served": true` and `"models_used": { "target_name": "live_prediction_db" }`.

### B. Real-Time ML Fallback Mode
If no live prediction database has been generated/loaded yet:
1. The server loads the individual model pickle files and performs real-time model inference.
2. Under heavy load, it routes predictions through a request queue with worker limits.

---

## 5. CLI Management

You can also trigger dataset generation manually from the CLI inside the project directory:

- **Regenerate Map Recommendations**:
  ```bash
  ./run.sh map-data
  ```
