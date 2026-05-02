from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

from .embeddings import DEFAULT_EMBEDDING_MODEL, add_track_embeddings, load_sentence_model
from .ensemble import WeightedSoftVotingEnsemble, select_weighted_models
from .schema import IGNORED_MODEL_FEATURES, PUBLIC_INPUT_COLUMNS, prepare_model_records
from .tabm_model import TabMClassifier


RANDOM_STATE = 21
DATASET_FILES = (
    ("dataset-of-60s.csv", "60"),
    ("dataset-of-70s.csv", "70"),
    ("dataset-of-80s.csv", "80"),
    ("dataset-of-90s.csv", "90"),
    ("dataset-of-00s.csv", "0"),
    ("dataset-of-10s.csv", "10"),
)


@dataclass
class TrainingRun:
    validation_scores: dict[str, dict[str, float]]
    test_scores: dict[str, dict[str, float]]
    selected_weights: dict[str, float]
    artifact_path: Path


def train_and_save_ensemble(
    *,
    data_dir: str | Path = "data",
    artifact_path: str | Path = "music-classifier/app/hit_ensemble.joblib",
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    feature_cache_path: str | Path | None = "artifacts/training_features.joblib",
    rebuild_feature_cache: bool = False,
    candidate_models: Sequence[str] | None = None,
    tabm_epochs: int = 80,
    tabm_device: str = "auto",
    tree_device: str = "auto",
    verbose: bool = False,
) -> TrainingRun:
    import joblib
    from sklearn.model_selection import train_test_split

    emit = _make_logger(verbose)
    emit("Loading training data")
    df = load_training_dataframe(data_dir)
    emit(f"Loaded {len(df)} rows")
    X, y = build_training_features(
        df,
        embedding_model_name=embedding_model_name,
        feature_cache_path=feature_cache_path,
        rebuild_feature_cache=rebuild_feature_cache,
        verbose=verbose,
    )

    emit("Splitting train/validation/test sets")
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        train_size=0.8,
        shuffle=True,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        train_size=0.75,
        shuffle=True,
        random_state=RANDOM_STATE,
        stratify=y_train_valid,
    )

    fitted_candidates = fit_candidate_models(
        X_train,
        y_train,
        include=candidate_models,
        tabm_epochs=tabm_epochs,
        tabm_device=tabm_device,
        tree_device=tree_device,
        verbose=verbose,
    )
    validation_scores = {
        name: evaluate_classifier(model, X_valid, y_valid)
        for name, model in fitted_candidates.items()
    }
    emit("Validation scores")
    for name, scores in validation_scores.items():
        emit(f"  {name}: roc_auc={scores['roc_auc']:.4f}")

    validation_auc = {
        name: scores["roc_auc"]
        for name, scores in validation_scores.items()
    }
    selected_weights = select_weighted_models(validation_auc)

    selected_validation_models = {
        name: fitted_candidates[name]
        for name in selected_weights
    }
    validation_ensemble = WeightedSoftVotingEnsemble(
        selected_validation_models,
        selected_weights,
        metadata={"validation_scores": validation_scores},
    )
    test_scores = {
        **{
            name: evaluate_classifier(model, X_test, y_test)
            for name, model in selected_validation_models.items()
        },
        "weighted_ensemble": evaluate_classifier(validation_ensemble, X_test, y_test),
    }

    emit(f"Selected weights: {selected_weights}")
    emit("Refitting selected models on the full dataset")
    final_models = fit_candidate_models(
        X,
        y,
        include=tuple(selected_weights),
        tabm_epochs=tabm_epochs,
        tabm_device=tabm_device,
        tree_device=tree_device,
        verbose=verbose,
    )
    artifact = WeightedSoftVotingEnsemble(
        models=final_models,
        weights=selected_weights,
        metadata={
            "validation_scores": validation_scores,
            "test_scores": test_scores,
            "embedding_model": embedding_model_name,
            "ignored_features": list(IGNORED_MODEL_FEATURES),
            "feature_cache_path": str(feature_cache_path) if feature_cache_path else None,
            "candidate_models": list(fitted_candidates),
            "tabm_epochs": tabm_epochs,
            "tabm_device": tabm_device,
            "tree_device": tree_device,
        },
    )

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    emit(f"Saved ensemble artifact to {artifact_path}")

    return TrainingRun(
        validation_scores=validation_scores,
        test_scores=test_scores,
        selected_weights=selected_weights,
        artifact_path=artifact_path,
    )


def load_training_dataframe(data_dir: str | Path):
    data_dir = Path(data_dir)
    try:
        import polars as pl

        frames = []
        for filename, decade in DATASET_FILES:
            frames.append(
                pl.read_csv(data_dir / filename).with_columns(
                    pl.lit(decade).alias("decade_of_release")
                )
            )
        return pl.concat(frames, how="vertical").to_pandas()
    except ImportError:
        import pandas as pd

        frames = []
        for filename, decade in DATASET_FILES:
            frame = pd.read_csv(data_dir / filename)
            frame["decade_of_release"] = decade
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def build_training_features(
    df: Any,
    *,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    feature_cache_path: str | Path | None = None,
    rebuild_feature_cache: bool = False,
    verbose: bool = False,
):
    import joblib
    import pandas as pd

    emit = _make_logger(verbose)
    if feature_cache_path and not rebuild_feature_cache:
        cached = _load_feature_cache(feature_cache_path, embedding_model_name)
        if cached is not None:
            emit(f"Loaded training features from {feature_cache_path}")
            return cached

    df = df.drop_duplicates(subset=["uri"]).reset_index(drop=True)
    records = df.loc[:, list(PUBLIC_INPUT_COLUMNS)].to_dict(orient="records")
    prepared = prepare_model_records(records)
    emit(f"Building track embeddings for {len(prepared)} records")
    sentence_model = load_sentence_model(embedding_model_name)
    feature_rows = add_track_embeddings(prepared, sentence_model)
    X = pd.DataFrame(feature_rows)
    y = df["target"].astype(int)

    if feature_cache_path:
        feature_cache_path = Path(feature_cache_path)
        feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "embedding_model": embedding_model_name,
                "ignored_features": list(IGNORED_MODEL_FEATURES),
                "X": X,
                "y": y,
            },
            feature_cache_path,
        )
        emit(f"Saved training features to {feature_cache_path}")

    return X, y


def create_preprocessor(X: Any):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

    categorical_cols = ["artist", "decade_of_release"]
    numerical_cols = [column for column in X.columns if column not in categorical_cols]

    return ColumnTransformer(
        [
            (
                "artist_encoder",
                TargetEncoder(smooth="auto", random_state=RANDOM_STATE),
                ["artist"],
            ),
            (
                "decade_encoder",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                ["decade_of_release"],
            ),
            ("scaler", StandardScaler(), numerical_cols),
        ],
        remainder="drop",
    )


def fit_candidate_models(
    X: Any,
    y: Any,
    *,
    include: Sequence[str] | None = None,
    tabm_epochs: int = 80,
    tabm_device: str = "auto",
    tree_device: str = "auto",
    verbose: bool = False,
):
    from sklearn.pipeline import Pipeline

    known_models = ("xgb", "catboost", "logreg", "tabm")
    names = tuple(include or known_models)
    unknown = sorted(set(names) - set(known_models))
    if unknown:
        known = ", ".join(known_models)
        raise ValueError(f"Unknown candidate models: {unknown}. Known models: {known}")

    use_tree_gpu = (
        _resolve_tree_gpu(tree_device)
        if any(name in names for name in ("xgb", "catboost"))
        else False
    )
    xgb_params: dict[str, Any] = {}
    catboost_params: dict[str, Any] = {}
    if use_tree_gpu:
        xgb_params["device"] = "cuda"
        catboost_params["task_type"] = "GPU"

    candidate_factories = {}
    if "xgb" in names:
        import xgboost as xgb

        candidate_factories["xgb"] = lambda: xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=0.028929893320248787,
            max_depth=10,
            subsample=0.5388792823570937,
            colsample_bytree=0.33367343724613546,
            min_child_weight=1,
            random_state=RANDOM_STATE,
            **xgb_params,
        )

    if "catboost" in names:
        from catboost import CatBoostClassifier

        candidate_factories["catboost"] = lambda: CatBoostClassifier(
            learning_rate=0.08765108142833057,
            depth=7,
            colsample_bylevel=0.4159606439575746,
            min_data_in_leaf=1,
            logging_level="Silent",
            random_state=RANDOM_STATE,
            **catboost_params,
        )

    if "logreg" in names:
        from sklearn.linear_model import LogisticRegression

        candidate_factories["logreg"] = lambda: LogisticRegression(
            C=0.05708097483824219,
            l1_ratio=1.0,
            solver="saga",
            max_iter=50000,
            random_state=RANDOM_STATE,
        )

    if "tabm" in names:
        candidate_factories["tabm"] = lambda: TabMClassifier(
            random_state=RANDOM_STATE,
            epochs=tabm_epochs,
            device=tabm_device,
            verbose=verbose,
        )

    emit = _make_logger(verbose)
    fitted = {}
    for name in names:
        emit(f"Fitting {name}")
        model = Pipeline(
            [
                ("preprocessor", create_preprocessor(X)),
                ("model", candidate_factories[name]()),
            ]
        )
        model.fit(X, y)
        fitted[name] = model
    return fitted


def evaluate_classifier(model: Any, X: Any, y: Any) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    probabilities = model.predict_proba(X)
    if hasattr(probabilities, "tolist"):
        probabilities = probabilities.tolist()
    positive = [row[1] for row in probabilities]
    predictions = [int(value >= 0.5) for value in positive]

    return {
        "roc_auc": float(roc_auc_score(y, positive)),
        "accuracy": float(accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions)),
    }


def _load_feature_cache(
    feature_cache_path: str | Path,
    embedding_model_name: str,
):
    import joblib

    feature_cache_path = Path(feature_cache_path)
    if not feature_cache_path.exists():
        return None

    cache = joblib.load(feature_cache_path)
    if cache.get("embedding_model") != embedding_model_name:
        return None
    if tuple(cache.get("ignored_features", ())) != IGNORED_MODEL_FEATURES:
        return None
    return cache["X"], cache["y"]


def _make_logger(verbose: bool) -> Callable[[str], None]:
    return print if verbose else lambda message: None


def _resolve_tree_gpu(tree_device: str) -> bool:
    requested = tree_device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("tree_device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return False

    import torch

    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("tree_device='cuda' was requested, but CUDA is not available")
    return cuda_available
