from litestar.testing import TestClient

from spotify_prediction.api import create_app


class FakeModelService:
    model_version = "test-model"
    feature_count = 21

    @property
    def model_loaded(self) -> bool:
        return True

    @property
    def embedding_model_loaded(self) -> bool:
        return True

    def predict_records(self, records: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {
                "track": str(record["track"]),
                "prediction": 1,
                "probabilities": [0.12, 0.88],
                "model_version": self.model_version,
            }
            for record in records
        ]


def _payload(track: str = "Hey Jude") -> dict[str, object]:
    return {
        "track": track,
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


def test_predict_accepts_track_and_returns_model_version() -> None:
    app = create_app(model_service=FakeModelService())

    with TestClient(app=app) as client:
        response = client.post("/predict", json=_payload())

    assert response.status_code == 201
    assert response.json() == {
        "track": "Hey Jude",
        "prediction": 1,
        "probabilities": [0.12, 0.88],
        "model_version": "test-model",
    }


def test_batch_predict_returns_result_per_track() -> None:
    app = create_app(model_service=FakeModelService())

    with TestClient(app=app) as client:
        response = client.post(
            "/batch_predict",
            json={"items": [_payload("Hey Jude"), _payload("Let It Be")]},
        )

    assert response.status_code == 201
    assert [item["track"] for item in response.json()["results"]] == ["Hey Jude", "Let It Be"]


def test_health_reports_loaded_resources() -> None:
    app = create_app(model_service=FakeModelService())

    with TestClient(app=app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "OK",
        "model_loaded": True,
        "embedding_model_loaded": True,
        "model_version": "test-model",
        "feature_count": 21,
    }
