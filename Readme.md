<div align="center">

# Spotify Hit Prediction

**Predicting Spotify hits using machine learning**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![Litestar](https://img.shields.io/badge/Litestar-2.21-green.svg)](https://litestar.dev)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-orange.svg)](https://catboost.ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

<img src="assets/spotify_logo.png" alt="Spotify Logo" width="200"/>

</div>

## About The Project

This project predicts whether a Spotify track is likely to be a hit using historical track metadata from 1960 through 2019. It includes a research notebook, a production training pipeline, a saved model artifact, and a Litestar API for serving predictions.

The runtime target is **CPython 3.13**.

## Project Structure

```text
Spotify_prediction/
├── src/spotify_prediction/       # Litestar API, feature builder, model service
├── spotify_hit_model/            # Training/ensemble classes needed by hit_ensemble.joblib
├── music-classifier/             # Docker serving config and model artifact location
├── data/                         # Kaggle Spotify hit predictor CSV files
├── Spotify_prediction.ipynb      # Research notebook
├── pipeline_generator.py         # Canonical production artifact generator
├── Research_report.md            # Current research and model metrics
├── pyproject.toml                # Python 3.13 dependency groups
└── tests/                        # Lightweight unit/API tests
```

`spotify_hit_model` remains in the repository because the current `hit_ensemble.joblib` artifact is serialized with classes from that package.

## Local Development

```bash
python -V  # expected: Python 3.13.x
uv run --extra dev pytest -q -s
```

Open the research notebook with:

```bash
jupyter notebook Spotify_prediction.ipynb
```

## Build The Production Artifact

The production artifact is `music-classifier/app/hit_ensemble.joblib`.

```bash
uv run --extra train python pipeline_generator.py \
  --hf-home .hf-cache \
  --offline-embeddings \
  --tabm-device cpu \
  --tree-device cpu \
  --models xgb catboost logreg
```

Omit `--offline-embeddings` on the first run if the `all-MiniLM-L6-v2` embedding model is not cached yet.

Verify the artifact before serving or committing it:

```bash
uv run --extra train python -c "import joblib; joblib.load('music-classifier/app/hit_ensemble.joblib'); print('ok')"
```

## Run The API

Local launch from the repository root:

```bash
pip install -r music-classifier/requirements.txt
PYTHONPATH=src:. uvicorn spotify_prediction.api:app --host 0.0.0.0 --port 8000
```

PowerShell:

```powershell
$env:PYTHONPATH = "src;."
uvicorn spotify_prediction.api:app --host 0.0.0.0 --port 8000
```

Docker:

```bash
cd music-classifier
docker compose up --build app
```

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Model and embedding-model status |
| POST | `/predict` | Single track prediction |
| POST | `/batch_predict` | Batch prediction |

Example request:

```json
{
  "track": "Hey Jude",
  "artist": "The Beatles",
  "decade_of_release": 1968,
  "danceability": 0.5,
  "energy": 0.7,
  "key": 7,
  "loudness": -8.5,
  "mode": 1,
  "speechiness": 0.03,
  "acousticness": 0.2,
  "instrumentalness": 0.0,
  "liveness": 0.1,
  "valence": 0.8,
  "tempo": 120.0,
  "duration_ms": 431000,
  "time_signature": 4,
  "chorus_hit": 0.5,
  "sections": 8
}
```

Unknown request fields are rejected by the API.

## Current Model

The current artifact selected CatBoost with weight `1.0` from the weighted-ensemble out-of-fold selection process.

| Metric | Value |
| --- | ---: |
| 5-fold CV ROC-AUC | 0.9949 |
| Holdout ROC-AUC | 0.9958 |
| Holdout accuracy | 0.9707 |
| Holdout F1 | 0.9705 |

## References

- [Research report](Research_report.md)
- [API and Docker notes](music-classifier/README.md)
- [Training entry point](pipeline_generator.py)
