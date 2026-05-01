# Music Classifier API

Litestar API for Spotify hit prediction.

## Run With Docker

```bash
docker compose up --build app
```

The image installs the `api` dependency extra, including the model runtime and SentenceTransformer stack. This is intentional: heavy serving dependencies live inside the container instead of the lightweight local test environment.

## Retrain In Docker

```bash
docker compose --profile train run --build trainer
```

The trainer installs the `train` extra, runs `spotify-train`, and writes `app/model.joblib` plus `app/model.metadata.json`.

## Local Smoke Test

After the API starts:

```bash
python test_api.py
```

## Endpoints

- `GET /health`
- `POST /predict`
- `POST /batch_predict`

`/predict` and `/batch_predict` accept track names, not precomputed embeddings. The service computes embeddings internally and then calls the saved classifier artifact.
