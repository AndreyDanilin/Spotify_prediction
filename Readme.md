<div align="center">

# Spotify Hit Prediction

Predict Spotify hits with a reproducible Polars + ML pipeline and a Litestar API.

</div>

## What Changed In v2

- API migrated from FastAPI to **Litestar**.
- Data loading and feature assembly are **Polars-first**.
- The API accepts `track` and computes text embeddings internally; clients no longer need to send `track_emb`.
- `pipeline_generator.py` now delegates to a model-selection runner that compares candidates by ROC-AUC and saves the best artifact.
- Dependencies are managed with `pyproject.toml` and uv extras:
  - `dev` for lightweight tests.
  - `api` for serving inside Docker.
  - `train` for full retraining and ensemble experiments.
  - `notebook` for notebook execution.

## Project Structure

```text
Spotify_prediction/
├── src/spotify_prediction/       # Package: features, training, model service, Litestar API
├── music-classifier/             # Docker/API wrapper and smoke test
├── data/                         # Kaggle Spotify hit predictor CSV files
├── Spotify_prediction.ipynb      # Research notebook with v2 ensemble section
├── pipeline_generator.py         # Compatibility training entrypoint
├── pyproject.toml                # Python 3.12 dependency groups
└── tests/                        # Lightweight unit/API tests
```

## Local Development

```bash
uv run --extra dev pytest -q -s
```

The `-s` flag avoids a pytest capture issue on some WSL-mounted Windows paths.

## Train The Model

Full training uses the heavier ML stack:

```bash
uv run --extra train spotify-train --data-dir data --output music-classifier/app/model.joblib
```

The runner:

1. Loads decade CSVs with Polars.
2. Drops duplicate Spotify URIs.
3. Builds SentenceTransformer embeddings for track names.
4. Adds compact audio interaction features.
5. Compares candidate single models and a soft-voting ensemble by stratified CV ROC-AUC.
6. Saves `model.joblib` and `model.metadata.json`.

## Run The API

```bash
uv run --extra api uvicorn spotify_prediction.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Model and embedding status |
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

## Docker

Serving dependencies are intentionally kept inside the image:

```bash
cd music-classifier
docker compose up --build app
```

To retrain in a container profile:

```bash
cd music-classifier
docker compose --profile train run --build trainer
```

This split keeps local and CI feedback light while still giving a reproducible heavy ML environment when needed.

## Research

`Spotify_prediction.ipynb` keeps the historical analysis and adds a final v2 Polars-first ensemble section. The production trainer and notebook now share the same package functions, so research and serving do not drift as easily.
