import numpy as np

from spotify_prediction.model_service import ModelService


class FakeEmbeddingModel:
    def encode(self, texts: list[str], show_progress_bar: bool = False):
        assert texts == ["Hey Jude"]
        assert show_progress_bar is False
        return np.array([[0.1, 0.2, 0.3]], dtype=float)


class FakePreprocessor:
    feature_names_in_ = ["artist", "track_emb_0", "track_emb_1", "track_emb_2"]


class FakeListProbabilityModel:
    model_version = "2.0"
    preprocessor = FakePreprocessor()

    def predict(self, records):
        return [1]

    def predict_proba(self, records):
        return [[0.12, 0.88]]


class NoLoadEmbeddingService(ModelService):
    def __init__(self) -> None:
        super().__init__("unused.joblib")
        self.embedding_accessed = False

    @property
    def embedding_model(self):
        self.embedding_accessed = True
        raise RuntimeError("embedding model should not be loaded by health")


def _payload() -> dict[str, object]:
    return {
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
        "sections": 8,
    }


def test_model_service_handles_list_probabilities_and_artifact_metadata() -> None:
    service = ModelService("unused.joblib")
    service._model = FakeListProbabilityModel()
    service._embedding_model = FakeEmbeddingModel()

    result = service.predict_records([_payload()])

    assert result == [
        {
            "track": "Hey Jude",
            "prediction": 1,
            "probabilities": [0.12, 0.88],
            "model_version": "2.0",
        }
    ]
    assert service.feature_count == 4


def test_model_version_falls_back_to_path_stem_when_model_cannot_load() -> None:
    service = ModelService("missing-model.joblib")

    assert service.model_loaded is False
    assert service.model_version == "missing-model"


def test_embedding_loaded_status_does_not_trigger_lazy_load() -> None:
    service = NoLoadEmbeddingService()

    assert service.embedding_model_loaded is False
    assert service.embedding_accessed is False
