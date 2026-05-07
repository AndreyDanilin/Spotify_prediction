FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY music-classifier/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY spotify_hit_model/ ./spotify_hit_model/
COPY music-classifier/app/ ./app/
COPY music-classifier/scripts/ ./scripts/

RUN mkdir -p ./music-classifier \
    && cp -r ./app ./music-classifier/app

CMD ["uvicorn", "spotify_prediction.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
