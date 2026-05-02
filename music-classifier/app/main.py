from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

for candidate in (Path.cwd(), *Path(__file__).resolve().parents):
    if (candidate / "spotify_hit_model").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from config import MODEL_PATH, settings
from spotify_hit_model.embeddings import build_inference_frame, load_sentence_model
from spotify_hit_model.schema import prepare_model_records


pipeline = None
sentence_model = None


@asynccontextmanager
async def lifespan(app):
    global pipeline, sentence_model

    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"Pipeline successfully loaded from {MODEL_PATH}")

        sentence_model = load_sentence_model(settings.EMBEDDING_MODEL)
        print(f"Sentence Transformer model loaded: {settings.EMBEDDING_MODEL}")
    except Exception as exc:
        print(f"Error loading models: {exc}")
        raise RuntimeError("Model loading failed") from exc

    yield

    print("Shutting down application...")


app = FastAPI(
    title="Music Track Classifier API",
    description="API for Spotify hit prediction with a weighted ensemble model",
    version=settings.MODEL_VERSION,
    lifespan=lifespan,
)


class TrackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artist: str
    track: str
    decade_of_release: str | int
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: int
    time_signature: int
    chorus_hit: float
    sections: int


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[TrackRequest]


@app.post("/predict")
async def predict(request: TrackRequest):
    try:
        frame = _build_prediction_frame([_dump_model(request)])
        probabilities = pipeline.predict_proba(frame)
        prediction = pipeline.predict(frame)
        probability_row = _probability_rows(probabilities)[0]

        return {
            "prediction": int(prediction[0]),
            "probabilities": probability_row,
            "model_version": _model_version(),
            "track_embedding_dim": _track_embedding_dim(frame),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


@app.post("/batch_predict")
async def batch_predict(request: BatchRequest):
    try:
        raw_items = [_dump_model(item) for item in request.items]
        frame = _build_prediction_frame(raw_items)
        predictions = pipeline.predict(frame)
        probabilities = _probability_rows(pipeline.predict_proba(frame))

        results = [
            {
                "track": raw_items[index]["track"],
                "prediction": int(predictions[index]),
                "probabilities": probabilities[index],
                "model_version": _model_version(),
            }
            for index in range(len(raw_items))
        ]
        return {"results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}") from exc


@app.get("/health")
async def health_check():
    return {
        "status": "OK",
        "model_loaded": hasattr(pipeline, "predict") and hasattr(pipeline, "predict_proba"),
        "embedding_model_loaded": hasattr(sentence_model, "encode"),
        "model_version": _model_version(),
    }


def _build_prediction_frame(raw_items):
    prepared_records = prepare_model_records(raw_items)
    return build_inference_frame(prepared_records, sentence_model)


def _dump_model(model: BaseModel) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _probability_rows(probabilities):
    if hasattr(probabilities, "tolist"):
        probabilities = probabilities.tolist()
    return [[float(value) for value in row] for row in probabilities]


def _track_embedding_dim(frame) -> int:
    return sum(1 for column in frame.columns if str(column).startswith("track_emb_"))


def _model_version() -> str:
    return str(getattr(pipeline, "model_version", settings.MODEL_VERSION))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=120,
    )
