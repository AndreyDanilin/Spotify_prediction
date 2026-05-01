import numpy as np
import polars as pl

from spotify_prediction.features import build_model_frame, normalize_decade


def _track_payload(track: str, decade: int | str = 1977) -> dict[str, object]:
    return {
        "track": track,
        "artist": "Queen",
        "decade_of_release": decade,
        "danceability": 0.41,
        "energy": 0.72,
        "key": 0,
        "loudness": -6.2,
        "mode": 1,
        "speechiness": 0.05,
        "acousticness": 0.22,
        "instrumentalness": 0.0,
        "liveness": 0.18,
        "valence": 0.44,
        "tempo": 144.5,
        "duration_ms": 355000,
        "time_signature": 4,
        "chorus_hit": 42.1,
        "sections": 12,
    }


def test_normalize_decade_accepts_years_and_existing_labels() -> None:
    assert normalize_decade(1965) == "60"
    assert normalize_decade("2008") == "0"
    assert normalize_decade("10") == "10"
    assert normalize_decade("unknown") == "unknown"


def test_build_model_frame_is_polars_first_and_expands_embeddings() -> None:
    records = [_track_payload("Bohemian Rhapsody"), _track_payload("Somebody To Love", "70")]
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)

    frame = build_model_frame(records, embeddings)

    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 2
    assert "track" not in frame.columns
    assert frame["decade_of_release"].to_list() == ["70", "70"]
    assert frame["track_emb_2"].to_list() == [0.3, 0.6]
    assert "energy_danceability" in frame.columns
    assert "log_duration_ms" in frame.columns
