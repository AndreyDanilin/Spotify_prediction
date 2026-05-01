#!/usr/bin/env sh
set -eu

uvicorn spotify_prediction.api:app \
  --host "${UVICORN_HOST:-0.0.0.0}" \
  --port "${UVICORN_PORT:-8000}" \
  --log-level "${LOG_LEVEL:-info}"
