from __future__ import annotations

from typing import Any

from litestar import Litestar, get, post
from litestar.exceptions import HTTPException

from spotify_prediction.model_service import ModelService
from spotify_prediction.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TrackInput,
)


def create_app(model_service: Any | None = None) -> Litestar:
    service = model_service or ModelService.from_environment()

    @get("/health")
    async def health() -> HealthResponse:
        model_loaded = service.model_loaded
        embedding_model_loaded = service.embedding_model_loaded
        try:
            feature_count = service.feature_count
        except Exception:
            feature_count = 0

        return HealthResponse(
            status="OK" if model_loaded and embedding_model_loaded else "DEGRADED",
            model_loaded=model_loaded,
            embedding_model_loaded=embedding_model_loaded,
            model_version=service.model_version,
            feature_count=feature_count,
        )

    @post("/predict", status_code=201)
    async def predict(data: TrackInput) -> PredictionResponse:
        try:
            result = service.predict_records([data.model_dump()])[0]
        except Exception as exc:  # pragma: no cover - exercised through integration/runtime
            raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
        return PredictionResponse(**result)

    @post("/batch_predict", status_code=201)
    async def batch_predict(data: BatchPredictionRequest) -> BatchPredictionResponse:
        try:
            results = service.predict_records([item.model_dump() for item in data.items])
        except Exception as exc:  # pragma: no cover - exercised through integration/runtime
            raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}") from exc
        return BatchPredictionResponse(results=[PredictionResponse(**item) for item in results])

    return Litestar(route_handlers=[health, predict, batch_predict])


app = create_app()
