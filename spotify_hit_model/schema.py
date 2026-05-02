from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


TEXT_FEATURES = ("artist", "track")
CATEGORICAL_FEATURES = ("decade_of_release",)
IGNORED_MODEL_FEATURES = ("speechiness", "instrumentalness")

NUMERIC_PUBLIC_FEATURES = (
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
)

PUBLIC_INPUT_COLUMNS = (
    "artist",
    "track",
    "decade_of_release",
    *NUMERIC_PUBLIC_FEATURES,
)

MODEL_INPUT_COLUMNS = tuple(
    column for column in PUBLIC_INPUT_COLUMNS if column not in IGNORED_MODEL_FEATURES
)

NUMERIC_MODEL_FEATURES = tuple(
    column for column in NUMERIC_PUBLIC_FEATURES if column not in IGNORED_MODEL_FEATURES
)


def normalize_decade(value: Any) -> str:
    if value in {"0", "10", "60", "70", "80", "90"}:
        return str(value)
    try:
        year = int(value)
    except (TypeError, ValueError):
        return str(value)

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


def prepare_model_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    public_columns = set(PUBLIC_INPUT_COLUMNS)
    required_columns = set(PUBLIC_INPUT_COLUMNS)

    for index, record in enumerate(records):
        keys = set(record)
        unknown = sorted(keys - public_columns)
        if unknown:
            raise ValueError(f"Unknown input fields at row {index}: {unknown}")

        missing = sorted(required_columns - keys)
        if missing:
            raise ValueError(f"Missing input fields at row {index}: {missing}")

        row: dict[str, Any] = {}
        for column in MODEL_INPUT_COLUMNS:
            value = record[column]
            if column == "decade_of_release":
                row[column] = normalize_decade(value)
            elif column in TEXT_FEATURES:
                row[column] = str(value)
            elif column in NUMERIC_MODEL_FEATURES:
                row[column] = _coerce_number(value, column, index)
            else:
                row[column] = value
        prepared.append(row)

    return prepared


def _coerce_number(value: Any, column: str, row_index: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Field {column!r} at row {row_index} must be numeric, got {value!r}"
        ) from exc
