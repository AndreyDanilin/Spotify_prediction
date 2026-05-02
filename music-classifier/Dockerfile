FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY music-classifier/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY spotify_hit_model/ ./spotify_hit_model/
COPY music-classifier/app/ ./app/
COPY music-classifier/scripts/ ./scripts/

WORKDIR /app/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
