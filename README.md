# Myanmar Agricultural Experimental Model Server

Private FastAPI inference service for the Myanmar Agriculture Intelligence project.
It remains a separate repository and process from the Node.js application backend.

```text
Website :3000 -> Node gateway :8000 -> Model server :8001 -> primary model artifacts
```

## 🛰️ GEE Live Data Pipeline & Prediction Database

To guarantee sub-millisecond response times in production, the server features a live ingestion pipeline and pre-computed predictions database:

1. **GEE Ingestion (`POST /api/v1/pipeline/update`)**: Allows uploading a monthly dynamic CSV export directly from Google Earth Engine.
2. **Background Ingestion Pipeline**: When a CSV is uploaded, the server runs a background task that:
   - Merges the new satellite observations with the static features dataset.
   - Batch predicts all 40 ML target outputs across all grid points.
   - Saves predictions to `live_predictions.parquet`.
   - Regenerates `map_recommendations.json` (ranking all 17 crops).
   - Hot-reloads the active prediction cache without server downtime.
3. **Instant Lookup Mode**: When the prediction database is active, `/predict` performs an $O(1)$ system index lookup or an $O(\log N)$ KD-Tree coordinate lookup, returning predictions in **<5ms** without running ML inference.

For detailed integration guides, see [Backend Engineer's Guide](file:///home/yan9htun/Desktop/gee/docs/backend_engineer_guide.md).

---

This release contains 40 primary Random Forest artifacts trained from real environmental,
satellite, soil, terrain, land-cover and infrastructure features. The training targets are
rule-engineered surrogate labels, not field-observed ground truth. Outputs are therefore
marked `experimental`, `rule_engineered_surrogate`, and `field_validated: false` in every
API response.

- 40/40 primary artifacts are present in the local `models/` directory.
- 31 models passed the repository's current diagnostic suite; 9 are flagged for review.
- Existing scores use random train/test splits, not spatial and temporal holdouts.
- The precipitation reconstruction is flagged for suspected target leakage.
- Models must not be presented as approved agronomic advice or production forecasts.
- Primary `.pkl` files are intentionally Git-ignored. A server deployment must mount or
  download the exact immutable artifacts declared in `models/manifest.json`.

The API publishes the artifacts that actually exist, without renaming them into planned
models. Suitability outputs use `poor/moderate/good/excellent`; the released health output
is only `crop_health_score` (0–1), and yield is `crop_yield_t_ha`. There are currently no
artifacts for `crop_health_status`, `crop_stress_type`, or `crop_damage_percent`.
`agricultural_gdp_forecast` is disclosed as a 0–1 surrogate index, not as a time-series
forecast or currency value. `GET /api/v1/models` is the authoritative contract.

## Correctness guarantees added to serving

- A verified 1,029,348-row spatial index is row-aligned with a compact 75-feature
  serving matrix; persistent DataFrame memory is reduced without changing feature order.
- Coordinate requests require an observation month and have an 8 km maximum match radius.
- Nearest cells are selected on the unit sphere, then verified with Haversine distance.
- Missing locations, data, models, checksums or model execution fail closed.
- No invented default feature row, modulo coordinate lookup, random prediction or silent
  prototype fallback is allowed.
- Released 500-tree estimators are never pruned or changed at runtime.
- Readiness and catalog responses verify all model/data SHA-256 digests before success.
- All-target inference is sequential and memory-bounded instead of retaining 40 models.
- Timed-out worker threads retain their capacity slot until they actually finish; one
  inference runs per process to keep large estimator references within the RAM budget.
- Cache keys include the API contract, model catalog and feature-data release versions.

## Local setup

Python 3.12 is required because the released artifacts were produced with
scikit-learn 1.9.0.

```bash
cd /Users/phyomyatmin/Desktop/GEO_MODEL_SERVER
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env
./run.sh serve
```

The model API is then available at `http://127.0.0.1:8001`.

```bash
curl http://127.0.0.1:8001/api/v1/ready
curl http://127.0.0.1:8001/api/v1/models
```

Run its integration tests independently:

```bash
.venv/bin/python -m pytest server/test_server.py
```

## Prediction API

`POST /api/v1/predict` requires exactly one verified locator and an explicit target list.

```json
{
  "request_id": "demo-001",
  "lat": 15.731919,
  "lon": 95.324433,
  "observation_month": "2018-01",
  "targets": ["heat_stress_risk", "crop_yield_t_ha"],
  "composite_features": []
}
```

An exact `sample_id` can replace `lat`, `lon` and `observation_month`. In local
development, `include_all_targets: true` can exercise all 40 outputs sequentially.
Production limits a synchronous request to 17 expanded targets so the bounded
crop-suitability tier request can run. All-40 inference remains a local audit
operation and requires a future durable asynchronous batch endpoint.

The current manifest is not field-validated or production-approved. Production
startup therefore refuses it by default. A hackathon/demo deployment must set
`ALLOW_EXPERIMENTAL_RELEASE=true` and retain the experimental labels; a real
production deployment requires a manifest with governance approval.

Each target response includes the value, uncalibrated tree-vote share when available, unit, model
version, artifact checksum, feature-schema checksum, diagnostic status and warnings.
The response also includes the matched cell, distance, observation period and row-level
data provenance.

Available endpoints:

- `POST /api/v1/predict`
- `GET /api/v1/models`
- `GET /api/v1/live`
- `GET /api/v1/ready`
- `GET /api/v1/health`
- `/docs` in local development only

The unsafe legacy `/boost` route has been removed.

## Rebuild verified serving metadata

When the QA-approved regional Parquet releases change, rebuild the locator table and then
the model catalog. The command refuses any row-order or shared-feature mismatch.

```bash
.venv/bin/python scripts/build_spatial_index.py \
  --source-root ../myanmar-agri-geo-csv-pipeline/data/output
.venv/bin/python scripts/generate_model_manifest.py
```

## Docker (local)

The Compose project exposes only the model API on host port 8001. Its Redis cache remains
private, so it does not conflict with the Node repository's Redis port.

```bash
docker compose up --build
```

Models and serving Parquet files are mounted read-only. The image runs as a non-root user,
uses one Uvicorn worker to avoid duplicating multi-gigabyte model memory, and disables
prototype/boost behavior.

Production configuration refuses disabled authentication, disabled startup checksum
verification, prototype models, or more than one inference per process:

```bash
docker compose -f docker-compose.yml -f docker-compose.production.yml up --build
```

## Node gateway configuration

For a native Node backend:

```env
MODEL_SERVER_URL=http://127.0.0.1:8001
```

For Node running in Docker Desktop while this Compose project is exposed locally:

```env
MODEL_SERVER_URL=http://host.docker.internal:8001
```

Set the same long `MODEL_SERVER_API_KEY` in both services and enable
`AUTH_REQUIRED=true` before a server deployment. In production, place this model service
on a private network; only the Node gateway should be publicly reachable.

## Remaining production gates

Production approval still requires field-verified labels, spatial and temporal holdout
evaluation, remediation/retraining of flagged models, an immutable artifact registry or
model OCI image, a durable asynchronous batch queue, load testing, monitoring, security
review and agronomist-approved action templates. The current serving path is hardened for
honest local integration; the model science is not yet production-approved.
