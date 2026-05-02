from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WeightedSoftVotingEnsemble:
    models: dict[str, Any]
    weights: dict[str, float]
    model_version: str = "2.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("At least one model is required")

        missing = sorted(set(self.models) - set(self.weights))
        if missing:
            raise ValueError(f"Missing weights for models: {missing}")

        extra = sorted(set(self.weights) - set(self.models))
        if extra:
            raise ValueError(f"Weights provided for unknown models: {extra}")

        total = sum(float(value) for value in self.weights.values())
        if total <= 0:
            raise ValueError("Model weights must sum to a positive value")

        self.weights = {name: float(value) / total for name, value in self.weights.items()}

    def predict_proba(self, records: Any) -> list[list[float]]:
        combined: list[list[float]] | None = None

        for name, model in self.models.items():
            raw_probabilities = model.predict_proba(records)
            probabilities = _to_probability_rows(raw_probabilities)
            weight = self.weights[name]

            if combined is None:
                combined = [[0.0, 0.0] for _ in probabilities]

            if len(probabilities) != len(combined):
                raise ValueError(
                    f"Model {name!r} returned {len(probabilities)} rows, expected "
                    f"{len(combined)}"
                )

            for row_index, row in enumerate(probabilities):
                if len(row) != 2:
                    raise ValueError(
                        f"Model {name!r} returned non-binary probabilities at row "
                        f"{row_index}: {row}"
                    )
                combined[row_index][0] += weight * float(row[0])
                combined[row_index][1] += weight * float(row[1])

        assert combined is not None
        return [[round(value, 12) for value in row] for row in combined]

    def predict(self, records: Any, threshold: float = 0.5) -> list[int]:
        return [int(row[1] >= threshold) for row in self.predict_proba(records)]


def select_weighted_models(
    validation_scores: dict[str, float],
    *,
    tolerance: float = 0.02,
    min_models: int = 2,
) -> dict[str, float]:
    if not validation_scores:
        raise ValueError("validation_scores must not be empty")

    ranked = sorted(validation_scores.items(), key=lambda item: item[1], reverse=True)
    best_score = ranked[0][1]
    selected = {
        name: score
        for name, score in ranked
        if score >= best_score - tolerance
    }

    for name, score in ranked[:min_models]:
        selected.setdefault(name, score)

    raw_weights = {
        name: max(score - 0.5, 1e-6)
        for name, score in selected.items()
    }
    total = sum(raw_weights.values())
    return {name: value / total for name, value in raw_weights.items()}


def _to_probability_rows(probabilities: Any) -> list[list[float]]:
    if hasattr(probabilities, "tolist"):
        probabilities = probabilities.tolist()
    return [list(row) for row in probabilities]
