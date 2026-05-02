from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from spotify_prediction.features import AUDIO_FEATURES, TARGET_COLUMN, build_model_frame

RANDOM_STATE = 21
DECADE_FILES = {
    "60": "dataset-of-60s.csv",
    "70": "dataset-of-70s.csv",
    "80": "dataset-of-80s.csv",
    "90": "dataset-of-90s.csv",
    "0": "dataset-of-00s.csv",
    "10": "dataset-of-10s.csv",
}


@dataclass(frozen=True)
class CandidateResult:
    name: str
    roc_auc_mean: float
    roc_auc_std: float
    estimator: Any


def choose_cv_splits(y: np.ndarray, requested_splits: int = 5) -> int:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Training data must contain at least two target classes")
    return max(2, min(int(requested_splits), int(counts.min())))


def load_decade_datasets(data_dir: str | Path = "data") -> pl.DataFrame:
    data_path = Path(data_dir)
    frames = []
    for decade, filename in DECADE_FILES.items():
        frame = pl.read_csv(data_path / filename).with_columns(pl.lit(decade).alias("decade_of_release"))
        frames.append(frame)
    return pl.concat(frames, how="vertical_relaxed").select(
        [
            "uri",
            "track",
            "artist",
            *AUDIO_FEATURES,
            "decade_of_release",
            TARGET_COLUMN,
        ]
    )


def drop_duplicate_tracks(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.unique(subset=["uri"], keep="first")


def encode_tracks(tracks: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(tracks, show_progress_bar=True, batch_size=64), dtype=float)


def build_training_frame(source: pl.DataFrame, embeddings: np.ndarray) -> tuple[pl.DataFrame, np.ndarray]:
    records = source.drop(TARGET_COLUMN).to_dicts()
    model_frame = build_model_frame(records, embeddings)
    return model_frame, source[TARGET_COLUMN].to_numpy()


def make_preprocessor(feature_names: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical = [column for column in ["artist", "decade_of_release"] if column in feature_names]
    numeric = [column for column in feature_names if column not in categorical]
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_candidates(feature_names: list[str]) -> dict[str, Any]:
    from catboost import CatBoostClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    def pipe(model: Any) -> Pipeline:
        return Pipeline([("preprocessor", make_preprocessor(feature_names)), ("model", model)])

    candidates = {
        "logreg": pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        "random_forest": pipe(RandomForestClassifier(n_estimators=350, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)),
        "extra_trees": pipe(ExtraTreesClassifier(n_estimators=350, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)),
        "xgboost": pipe(
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.035,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="auc",
                tree_method="hist",
                random_state=RANDOM_STATE,
            )
        ),
        "catboost": pipe(
            CatBoostClassifier(
                iterations=600,
                learning_rate=0.035,
                depth=6,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=RANDOM_STATE,
                verbose=False,
            )
        ),
    }
    candidates["soft_voting"] = VotingClassifier(
        estimators=[("xgboost", candidates["xgboost"]), ("catboost", candidates["catboost"]), ("extra_trees", candidates["extra_trees"])],
        voting="soft",
        n_jobs=-1,
    )
    return candidates


def evaluate_candidates(X, y: np.ndarray, candidates: dict[str, Any], requested_splits: int = 5) -> list[CandidateResult]:
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    splits = choose_cv_splits(y, requested_splits)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    results: list[CandidateResult] = []
    for name, estimator in candidates.items():
        scores = []
        for train_idx, valid_idx in cv.split(X, y):
            fold_estimator = clone(estimator)
            fold_estimator.fit(X.iloc[train_idx], y[train_idx])
            probabilities = fold_estimator.predict_proba(X.iloc[valid_idx])[:, 1]
            scores.append(roc_auc_score(y[valid_idx], probabilities))
        results.append(CandidateResult(name=name, roc_auc_mean=float(np.mean(scores)), roc_auc_std=float(np.std(scores)), estimator=estimator))
    return sorted(results, key=lambda result: result.roc_auc_mean, reverse=True)


def fit_best_model(X, y: np.ndarray, results: list[CandidateResult]) -> CandidateResult:
    best = results[0]
    best.estimator.fit(X, y)
    return best


def save_model(result: CandidateResult, output_path: str | Path, feature_count: int, all_results: list[CandidateResult]) -> None:
    import joblib

    model_path = Path(output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.estimator, model_path)
    metadata = {
        "model_version": datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S"),
        "selected_model": result.name,
        "roc_auc_mean": result.roc_auc_mean,
        "roc_auc_std": result.roc_auc_std,
        "feature_count": feature_count,
        "candidates": [
            {"name": item.name, "roc_auc_mean": item.roc_auc_mean, "roc_auc_std": item.roc_auc_std}
            for item in all_results
        ],
    }
    model_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_training(data_dir: str | Path, output_path: str | Path, requested_splits: int = 5) -> CandidateResult:
    source = drop_duplicate_tracks(load_decade_datasets(data_dir))
    embeddings = encode_tracks(source["track"].cast(pl.Utf8).to_list())
    model_frame, y = build_training_frame(source, embeddings)
    X = model_frame.to_pandas()
    candidates = make_candidates(list(X.columns))
    results = evaluate_candidates(X, y, candidates, requested_splits=requested_splits)
    best = fit_best_model(X, y, results)
    save_model(best, output_path, feature_count=len(X.columns), all_results=results)
    return best


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and select the best Spotify hit classifier.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="music-classifier/app/model.joblib")
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args(argv)
    best = run_training(args.data_dir, args.output, requested_splits=args.cv)
    print(f"Selected {best.name} with ROC-AUC={best.roc_auc_mean:.4f} (+/- {best.roc_auc_std:.4f})")


if __name__ == "__main__":
    main()
