FROM python:3.12-slim

ARG INSTALL_EXTRAS=api

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock Readme.md ./
COPY src ./src
COPY data ./data
COPY music-classifier/app ./music-classifier/app
COPY music-classifier/scripts ./music-classifier/scripts

RUN pip install --no-cache-dir ".[${INSTALL_EXTRAS}]"

EXPOSE 8000

CMD ["uvicorn", "spotify_prediction.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
