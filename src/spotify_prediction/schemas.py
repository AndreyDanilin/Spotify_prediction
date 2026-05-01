from __future__ import annotations

from pydantic import BaseModel, Field


class TrackInput(BaseModel):
    track: str = Field(min_length=1)
    artist: str = Field(min_length=1)
    decade_of_release: int | str
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


class BatchPredictionRequest(BaseModel):
    items: list[TrackInput] = Field(min_length=1)


class PredictionResponse(BaseModel):
    track: str
    prediction: int
    probabilities: list[float]
    model_version: str


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    embedding_model_loaded: bool
    model_version: str
    feature_count: int
