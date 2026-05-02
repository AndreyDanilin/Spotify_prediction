from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_inference_frame(
    records: Sequence[Mapping[str, Any]],
    sentence_model: Any,
):
    import pandas as pd

    rows = add_track_embeddings(records, sentence_model)
    return pd.DataFrame(rows)


def add_track_embeddings(
    records: Sequence[Mapping[str, Any]],
    sentence_model: Any,
) -> list[dict[str, Any]]:
    tracks = [str(record["track"]) for record in records]
    embeddings = sentence_model.encode(tracks, show_progress_bar=False)

    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()

    rows = []
    for record, embedding in zip(records, embeddings, strict=True):
        row = dict(record)
        row.pop("track", None)
        for index, value in enumerate(embedding):
            row[f"track_emb_{index}"] = float(value)
        rows.append(row)
    return rows


def load_sentence_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
