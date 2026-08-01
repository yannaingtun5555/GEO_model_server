FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY server/ ./server/
COPY models/ ./models/
COPY models_prototypes/ ./models_prototypes/
COPY data/combined/combined_dataset.csv ./data/combined/combined_dataset.csv

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MAX_LOADED_MODELS=4
ENV MAX_RAM_MB=2048
ENV REDIS_HOST=redis_cache
ENV REDIS_PORT=6379

EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
