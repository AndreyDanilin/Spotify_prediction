from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from spotify_prediction.features import build_model_frame, to_pandas_for_sklearn

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ModelService:
    """Loads the persisted classifier and turns API records into predictions."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path | None = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else self.model_path.with_suffix(".metadata.json")
        self.embedding_model_name = embedding_model_name
        self._model: Any | None = None
        self._embedding_model: Any | None = None
        self._metadata: dict[str, Any] = {}

    @classmethod
    def from_environment(cls) -> "ModelService":
        default_path = Path(__file__).resolve().parents[2] / "music-classifier" / "app" / "hit_ensemble.joblib"
        model_path = Path(os.getenv("MODEL_PATH", str(default_path)))
        metadata_path = os.getenv("MODEL_METADATA_PATH")
        embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        return cls(model_path=model_path, metadata_path=metadata_path, embedding_model_name=embedding_model)

    @property
    def model(self) -> Any:
        if self._model is None:
            import joblib

            self._model = joblib.load(self.model_path)
            if self.metadata_path.exists():
                self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return self._model

    @property
    def embedding_model(self) -> Any:
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    @property
    def model_loaded(self) -> bool:
        try:
            return hasattr(self.model, "predict")
        except Exception:
            return False

    @property
    def embedding_model_loaded(self) -> bool:
        return self._embedding_model is not None and hasattr(self._embedding_model, "encode")

    @property
    def model_version(self) -> str:
        if self._metadata:
            return str(self._metadata.get("model_version", "unknown"))
        try:
            version = getattr(self.model, "model_version", None)
        except Exception:
            version = None
        if version is not None:
            return str(version)
        return self.model_path.stem

    @property
    def feature_count(self) -> int:
        feature_names = self._feature_names()
        if feature_names is not None:
            return len(feature_names)
        return int(self._metadata.get("feature_count", 0))

    def _encode_tracks(self, records: list[dict[str, Any]]) -> np.ndarray:
        texts = [str(record["track"]) for record in records]
        return np.asarray(self.embedding_model.encode(texts, show_progress_bar=False), dtype=float)

    def _feature_names(self) -> list[str] | None:
        feature_names = getattr(self.model, "feature_names_in_", None)
        if feature_names is None:
            preprocessor = getattr(self.model, "preprocessor", None)
            feature_names = getattr(preprocessor, "feature_names_in_", None)
        if feature_names is None:
            return None
        return [str(name) for name in feature_names]

    def predict_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        embeddings = self._encode_tracks(records)
        frame = build_model_frame(records, embeddings)
        feature_names = self._feature_names()
        model_input = to_pandas_for_sklearn(frame, feature_names)

        predictions = self.model.predict(model_input)
        probabilities = self.model.predict_proba(model_input)
        return [
            {
                "track": str(record["track"]),
                "prediction": int(predictions[idx]),
                "probabilities": [float(value) for value in _as_list(probabilities[idx])],
                "model_version": self.model_version,
            }
            for idx, record in enumerate(records)
        ]


def _as_list(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)
