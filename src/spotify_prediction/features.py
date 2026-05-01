from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import log1p
from typing import Any

import numpy as np
import polars as pl

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
    "chorus_hit",
    "sections",
]
CATEGORICAL_FEATURES = ["artist", "decade_of_release"]
IDENTIFIER_COLUMNS = ["uri", "track"]
TARGET_COLUMN = "target"


def normalize_decade(value: object) -> str:
    """Normalize release year/decade to labels used by the training data."""
    if value is None:
        return "unknown"
    text = str(value).strip()
    if text in {"0", "10", "60", "70", "80", "90", "unknown"}:
        return text
    try:
        year = int(float(text))
    except (TypeError, ValueError):
        return text or "unknown"

    if 1960 <= year < 1970:
        return "60"
    if 1970 <= year < 1980:
        return "70"
    if 1980 <= year < 1990:
        return "80"
    if 1990 <= year < 2000:
        return "90"
    if 2000 <= year < 2010:
        return "0"
    if 2010 <= year < 2020:
        return "10"
    return "unknown"


def _records_to_frame(records: Iterable[dict[str, Any]]) -> pl.DataFrame:
    frame = pl.DataFrame(list(records), infer_schema_length=None)
    if frame.is_empty():
        raise ValueError("At least one track record is required")
    return frame


def _coerce_base_columns(frame: pl.DataFrame) -> pl.DataFrame:
    for column in CATEGORICAL_FEATURES:
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit("unknown").alias(column))

    for column in AUDIO_FEATURES:
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(0.0).alias(column))

    return frame.with_columns(
        pl.col("artist").cast(pl.Utf8, strict=False).fill_null("unknown_artist"),
        pl.col("decade_of_release")
        .map_elements(normalize_decade, return_dtype=pl.Utf8)
        .fill_null("unknown"),
        *[
            pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0).alias(column)
            for column in AUDIO_FEATURES
        ],
    )


def _add_audio_interactions(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col("energy") * pl.col("danceability")).alias("energy_danceability"),
        (pl.col("valence") * pl.col("danceability")).alias("valence_danceability"),
        (pl.col("energy") * pl.col("valence")).alias("energy_valence"),
        pl.col("duration_ms").map_elements(lambda value: log1p(max(float(value), 0.0)), return_dtype=pl.Float64).alias("log_duration_ms"),
        (pl.col("tempo") / 10.0).floor().alias("tempo_bin"),
    )


def _embedding_frame(embeddings: np.ndarray) -> pl.DataFrame:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array")
    return pl.DataFrame({f"track_emb_{idx}": embeddings[:, idx] for idx in range(embeddings.shape[1])})


def build_model_frame(records: Iterable[dict[str, Any]], embeddings: Sequence[Sequence[float]] | np.ndarray) -> pl.DataFrame:
    """Build the model input frame with Polars and expanded text embeddings."""
    frame = _coerce_base_columns(_records_to_frame(records))
    frame = _add_audio_interactions(frame)

    embedding_array = np.asarray(embeddings, dtype=float)
    if embedding_array.shape[0] != frame.height:
        raise ValueError("Number of embeddings must match number of records")

    keep_columns = [
        column
        for column in frame.columns
        if column not in {"track"} and column != TARGET_COLUMN
    ]
    return frame.select(keep_columns).hstack(_embedding_frame(embedding_array))


def to_pandas_for_sklearn(frame: pl.DataFrame, columns: Sequence[str] | None = None):
    """Convert the Polars frame at the sklearn boundary only."""
    selected = frame.select(list(columns)) if columns else frame
    try:
        return selected.to_pandas()
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install the api or train extra to enable pandas conversion") from exc
