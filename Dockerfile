FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --gid 10001 app && \
    useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY requirements.txt ./
RUN pip install --requirement requirements.txt

COPY server/ ./server/
COPY pipeline/ ./pipeline/
COPY models/manifest.json ./catalog/manifest.json

RUN mkdir -p /models /data && chown -R app:app /app /models /data

ENV ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    MODELS_DIR=/models \
    MODEL_MANIFEST_FILE=/app/catalog/manifest.json \
    FEATURE_DATA_FILE=/data/features_serving.parquet \
    SPATIAL_INDEX_FILE=/data/spatial_index.parquet \
    ALLOW_PROTOTYPE_MODELS=false \
    BOOST_MODE=false

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/live', timeout=3)"]

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
