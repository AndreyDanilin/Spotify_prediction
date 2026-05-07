# Music Classifier API

Litestar ASGI API for Spotify hit prediction. The service loads `app/hit_ensemble.joblib`, computes track-name embeddings with `all-MiniLM-L6-v2`, and returns hit/non-hit probabilities.

The supported runtime target is **CPython 3.13**.

## Project Structure

```text
music-classifier/
├── app/
│   └── hit_ensemble.joblib      # Production model artifact
├── scripts/
│   └── start.sh                 # Uvicorn launcher for spotify_prediction.api:app
├── test_api.py                  # Manual smoke test for a running API
├── requirements.txt             # Container/local serving dependencies
├── Dockerfile
├── docker-compose.yml
└── README.md
```

The active application object lives in `src/spotify_prediction/api.py`. The `music-classifier/app` directory is now only the model-artifact location kept for compatibility with existing deployment paths.

## Build Or Refresh The Model Artifact

Run from the repository root:

```bash
uv run --extra train python pipeline_generator.py \
  --hf-home .hf-cache \
  --offline-embeddings \
  --tabm-device cpu \
  --tree-device cpu \
  --models xgb catboost logreg
```

Omit `--offline-embeddings` on the first run if the embedding model is not cached yet. The script writes `music-classifier/app/hit_ensemble.joblib` and may reuse `artifacts/training_features.joblib`.

Verify the artifact before serving or committing it:

```bash
uv run --extra train python -c "import joblib; joblib.load('music-classifier/app/hit_ensemble.joblib'); print('ok')"
```

## Local Launch

From the repository root:

```bash
python -V  # expected: Python 3.13.x
pip install -r music-classifier/requirements.txt
PYTHONPATH=src:. uvicorn spotify_prediction.api:app --host 0.0.0.0 --port 8000
```

On PowerShell, use:

```powershell
$env:PYTHONPATH = "src;."
uvicorn spotify_prediction.api:app --host 0.0.0.0 --port 8000
```

## Docker Launch

```bash
cd music-classifier
docker compose up --build app
```

The image installs `requirements.txt`, copies `src/spotify_prediction`, `spotify_hit_model`, and `music-classifier/app/hit_ensemble.joblib`, then starts:

```bash
uvicorn spotify_prediction.api:app --host 0.0.0.0 --port 8000
```

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Model and embedding-model status |
| POST | `/predict` | Single track prediction |
| POST | `/batch_predict` | Batch prediction |

### Health Response

```json
{
  "status": "OK",
  "model_loaded": true,
  "embedding_model_loaded": true,
  "model_version": "2.0",
  "feature_count": 399
}
```

If model loading fails, `/health` still returns HTTP 200 with `"status": "DEGRADED"`.

### Prediction Request

```json
{
  "artist": "The Beatles",
  "track": "Hey Jude",
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

Unknown request fields are rejected.

### Prediction Response

```json
{
  "track": "Hey Jude",
  "prediction": 1,
  "probabilities": [0.12, 0.88],
  "model_version": "2.0"
}
```

## Current Model Metrics

The current artifact selected CatBoost with weight `1.0` from the weighted-ensemble selection process.

| Metric | Value |
| --- | ---: |
| 5-fold CV ROC-AUC | 0.9949 |
| Holdout ROC-AUC | 0.9958 |
| Holdout accuracy | 0.9707 |
| Holdout F1 | 0.9705 |
