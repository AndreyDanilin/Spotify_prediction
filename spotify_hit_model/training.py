from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any, Mapping

from .embeddings import DEFAULT_EMBEDDING_MODEL, add_track_embeddings, load_sentence_model
from .ensemble import PreprocessedWeightedSoftVotingEnsemble, WeightedSoftVotingEnsemble
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
DEFAULT_MODEL_PARAMS_PATH = Path("artifacts/model_params.json")
DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "xgb": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.028929893320248787,
        "max_depth": 10,
        "subsample": 0.5388792823570937,
        "colsample_bytree": 0.33367343724613546,
        "min_child_weight": 1,
    },
    "catboost": {
        "learning_rate": 0.08765108142833057,
        "depth": 7,
        "colsample_bylevel": 0.4159606439575746,
        "min_data_in_leaf": 1,
        "logging_level": "Silent",
    },
    "logreg": {
        "C": 0.05708097483824219,
        "l1_ratio": 1.0,
        "solver": "saga",
        "max_iter": 50000,
    },
    "tabm": {},
}


@dataclass
class TrainingRun:
    validation_scores: dict[str, dict[str, float]]
    test_scores: dict[str, dict[str, float]]
    selected_weights: dict[str, float]
    artifact_path: Path


@dataclass
class CrossValidationRun:
    fold_scores: dict[str, list[dict[str, float]]]
    summary_scores: dict[str, dict[str, float]]
    oof_probabilities: dict[str, list[float]]


@dataclass
class EnsembleSelection:
    selected_weights: dict[str, float]
    individual_scores: dict[str, float]
    ensemble_score: float


def train_and_save_ensemble(
    *,
    data_dir: str | Path = "data",
    artifact_path: str | Path = "music-classifier/app/hit_ensemble.joblib",
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    feature_cache_path: str | Path | None = "artifacts/training_features.joblib",
    rebuild_feature_cache: bool = False,
    candidate_models: Sequence[str] | None = None,
    model_params_path: str | Path | None = DEFAULT_MODEL_PARAMS_PATH,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    cv_splits: int = 5,
    ensemble_min_improvement: float = 0.001,
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
    resolved_model_params = resolve_model_params(model_params_path, model_params)

    emit("Splitting train/validation/test sets")
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        train_size=0.8,
        shuffle=True,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    cv_run = cross_validate_candidate_models(
        X_train_valid,
        y_train_valid,
        include=candidate_models,
        model_params=resolved_model_params,
        n_splits=cv_splits,
        tabm_epochs=tabm_epochs,
        tabm_device=tabm_device,
        tree_device=tree_device,
        verbose=verbose,
    )
    validation_scores = cv_run.summary_scores
    emit("Cross-validation scores")
    for name, scores in validation_scores.items():
        emit(f"  {name}: roc_auc={scores['roc_auc']:.4f} +/- {scores['roc_auc_std']:.4f}")

    ensemble_selection = select_ensemble_from_oof(
        cv_run.oof_probabilities,
        y_train_valid,
        min_improvement=ensemble_min_improvement,
    )
    selected_weights = ensemble_selection.selected_weights

    emit(f"Selected weights: {selected_weights}")
    emit("Fitting selected models on train/validation data for holdout scoring")
    selected_holdout_models = fit_candidate_models(
        X_train_valid,
        y_train_valid,
        include=tuple(selected_weights),
        model_params=resolved_model_params,
        tabm_epochs=tabm_epochs,
        tabm_device=tabm_device,
        tree_device=tree_device,
        verbose=verbose,
    )
    holdout_ensemble = WeightedSoftVotingEnsemble(
        selected_holdout_models,
        selected_weights,
        metadata={
            "validation_scores": validation_scores,
            "oof_ensemble_score": ensemble_selection.ensemble_score,
        },
    )
    test_scores = {
        **{
            name: evaluate_classifier(model, X_test, y_test)
            for name, model in selected_holdout_models.items()
        },
        "weighted_ensemble": evaluate_classifier(holdout_ensemble, X_test, y_test),
    }

    emit("Refitting selected models on the full dataset")
    final_preprocessor, final_models = fit_preprocessed_candidate_models(
        X,
        y,
        include=tuple(selected_weights),
        model_params=resolved_model_params,
        tabm_epochs=tabm_epochs,
        tabm_device=tabm_device,
        tree_device=tree_device,
        verbose=verbose,
    )
    artifact = PreprocessedWeightedSoftVotingEnsemble(
        preprocessor=final_preprocessor,
        models=final_models,
        weights=selected_weights,
        metadata={
            "validation_scores": validation_scores,
            "test_scores": test_scores,
            "model_params": resolved_model_params,
            "cv_splits": cv_splits,
            "oof_individual_scores": ensemble_selection.individual_scores,
            "oof_ensemble_score": ensemble_selection.ensemble_score,
            "embedding_model": embedding_model_name,
            "ignored_features": list(IGNORED_MODEL_FEATURES),
            "feature_cache_path": str(feature_cache_path) if feature_cache_path else None,
            "candidate_models": list(candidate_models or DEFAULT_MODEL_PARAMS),
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
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    tabm_epochs: int = 80,
    tabm_device: str = "auto",
    tree_device: str = "auto",
    verbose: bool = False,
):
    names = _normalize_model_names(include)

    use_tree_gpu = (
        _resolve_tree_gpu(tree_device)
        if any(name in names for name in ("xgb", "catboost"))
        else False
    )
    xgb_device_params: dict[str, Any] = {}
    catboost_device_params: dict[str, Any] = {}
    if use_tree_gpu:
        xgb_device_params["device"] = "cuda"
        catboost_device_params["task_type"] = "GPU"

    candidate_factories = build_model_factories(
        include=names,
        model_params=model_params,
        tabm_epochs=tabm_epochs,
        tabm_device=tabm_device,
        verbose=verbose,
        device_params={
            "xgb": xgb_device_params,
            "catboost": catboost_device_params,
        },
    )

    emit = _make_logger(verbose)
    fitted = {}
    for name in names:
        from sklearn.pipeline import Pipeline

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


def fit_preprocessed_candidate_models(
    X: Any,
    y: Any,
    *,
    include: Sequence[str] | None = None,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    tabm_epochs: int = 80,
    tabm_device: str = "auto",
    tree_device: str = "auto",
    verbose: bool = False,
):
    names = _normalize_model_names(include)
    use_tree_gpu = (
        _resolve_tree_gpu(tree_device)
        if any(name in names for name in ("xgb", "catboost"))
        else False
    )
    xgb_device_params: dict[str, Any] = {}
    catboost_device_params: dict[str, Any] = {}
    if use_tree_gpu:
        xgb_device_params["device"] = "cuda"
        catboost_device_params["task_type"] = "GPU"

    candidate_factories = build_model_factories(
        include=names,
        model_params=model_params,
        tabm_epochs=tabm_epochs,
        tabm_device=tabm_device,
        verbose=verbose,
        device_params={
            "xgb": xgb_device_params,
            "catboost": catboost_device_params,
        },
    )
    emit = _make_logger(verbose)
    preprocessor = create_preprocessor(X)
    transformed = preprocessor.fit_transform(X, y)
    fitted = {}
    for name in names:
        emit(f"Fitting {name}")
        model = candidate_factories[name]()
        model.fit(transformed, y)
        fitted[name] = model
    return preprocessor, fitted


def build_model_factories(
    *,
    include: Sequence[str] | None = None,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    tabm_epochs: int = 80,
    tabm_device: str = "auto",
    verbose: bool = False,
    device_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Callable[[], Any]]:
    names = _normalize_model_names(include)
    params_by_model = resolve_model_params(None, model_params)
    device_params = device_params or {}
    factories: dict[str, Callable[[], Any]] = {}

    if "xgb" in names:
        import xgboost as xgb

        params = {
            **params_by_model["xgb"],
            "random_state": RANDOM_STATE,
            **dict(device_params.get("xgb", {})),
        }
        factories["xgb"] = lambda params=params: xgb.XGBClassifier(**params)

    if "catboost" in names:
        from catboost import CatBoostClassifier

        params = {
            **params_by_model["catboost"],
            "random_state": RANDOM_STATE,
            **dict(device_params.get("catboost", {})),
        }
        factories["catboost"] = lambda params=params: CatBoostClassifier(**params)

    if "logreg" in names:
        from sklearn.linear_model import LogisticRegression

        params = {**params_by_model["logreg"], "random_state": RANDOM_STATE}
        factories["logreg"] = lambda params=params: LogisticRegression(**params)

    if "tabm" in names:
        params = {
            **params_by_model["tabm"],
            "random_state": RANDOM_STATE,
            "epochs": tabm_epochs,
            "device": tabm_device,
            "verbose": verbose,
        }
        factories["tabm"] = lambda params=params: TabMClassifier(**params)

    return factories


def cross_validate_candidate_models(
    X: Any,
    y: Any,
    *,
    include: Sequence[str] | None = None,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    n_splits: int = 5,
    tabm_epochs: int = 80,
    tabm_device: str = "auto",
    tree_device: str = "auto",
    verbose: bool = False,
) -> CrossValidationRun:
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    names = _normalize_model_names(include)
    y_array = np.asarray(y, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_scores: dict[str, list[dict[str, float]]] = {name: [] for name in names}
    oof_probabilities: dict[str, list[float]] = {
        name: [0.0 for _ in range(len(y_array))]
        for name in names
    }

    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X, y_array), start=1):
        emit = _make_logger(verbose)
        emit(f"CV fold {fold_index}/{n_splits}")
        X_train = _take_rows(X, train_idx)
        X_valid = _take_rows(X, valid_idx)
        y_train = y_array[train_idx]
        y_valid = y_array[valid_idx]
        fitted = fit_candidate_models(
            X_train,
            y_train,
            include=names,
            model_params=model_params,
            tabm_epochs=tabm_epochs,
            tabm_device=tabm_device,
            tree_device=tree_device,
            verbose=verbose,
        )
        for name, model in fitted.items():
            probabilities = model.predict_proba(X_valid)
            positive = np.asarray(probabilities)[:, 1]
            for row_index, probability in zip(valid_idx, positive):
                oof_probabilities[name][int(row_index)] = float(probability)
            fold_scores[name].append(evaluate_classifier(model, X_valid, y_valid))

    return CrossValidationRun(
        fold_scores=fold_scores,
        summary_scores=_summarize_fold_scores(fold_scores),
        oof_probabilities=oof_probabilities,
    )


def select_ensemble_from_oof(
    oof_probabilities: Mapping[str, Sequence[float]],
    y: Any,
    *,
    min_improvement: float = 0.001,
    weight_grid_step: float = 0.05,
) -> EnsembleSelection:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    if not oof_probabilities:
        raise ValueError("oof_probabilities must not be empty")

    y_array = np.asarray(y, dtype=int)
    probability_arrays = {
        name: np.asarray(values, dtype=float)
        for name, values in oof_probabilities.items()
    }
    individual_scores = {
        name: float(roc_auc_score(y_array, probabilities))
        for name, probabilities in probability_arrays.items()
    }
    ranked = sorted(individual_scores, key=individual_scores.get, reverse=True)
    selected = [ranked[0]]
    selected_weights = {ranked[0]: 1.0}
    ensemble_probabilities = probability_arrays[ranked[0]].copy()
    ensemble_score = individual_scores[ranked[0]]

    candidate_weights = np.arange(weight_grid_step, 1.0, weight_grid_step)
    for candidate in ranked[1:]:
        best_candidate_score = ensemble_score
        best_candidate_weight = 0.0
        candidate_probabilities = probability_arrays[candidate]
        for candidate_weight in candidate_weights:
            mixed = (1.0 - candidate_weight) * ensemble_probabilities
            mixed += candidate_weight * candidate_probabilities
            score = float(roc_auc_score(y_array, mixed))
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate_weight = float(candidate_weight)

        if best_candidate_score >= ensemble_score + min_improvement:
            selected_weights = {
                name: weight * (1.0 - best_candidate_weight)
                for name, weight in selected_weights.items()
            }
            selected_weights[candidate] = best_candidate_weight
            selected.append(candidate)
            ensemble_probabilities = (
                (1.0 - best_candidate_weight) * ensemble_probabilities
                + best_candidate_weight * candidate_probabilities
            )
            ensemble_score = best_candidate_score

    total = sum(selected_weights.values())
    selected_weights = {name: value / total for name, value in selected_weights.items()}
    return EnsembleSelection(
        selected_weights=selected_weights,
        individual_scores=individual_scores,
        ensemble_score=ensemble_score,
    )


def load_model_params(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if "models" in raw:
        raw = raw["models"]
    return sanitize_model_params(raw)


def save_model_params(
    params: Mapping[str, Mapping[str, Any]],
    path: str | Path = DEFAULT_MODEL_PARAMS_PATH,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_model_params(params)
    with path.open("w", encoding="utf-8") as file:
        json.dump({"models": sanitized}, file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def resolve_model_params(
    path: str | Path | None = DEFAULT_MODEL_PARAMS_PATH,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    params = {
        name: dict(values)
        for name, values in DEFAULT_MODEL_PARAMS.items()
    }
    if path is not None and Path(path).exists():
        loaded = load_model_params(path)
        for name, values in loaded.items():
            params.setdefault(name, {}).update(values)
    if overrides:
        for name, values in overrides.items():
            params.setdefault(name, {}).update(dict(values))
    return sanitize_model_params(params)


def sanitize_model_params(
    params: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    sanitized = {
        name: dict(values)
        for name, values in params.items()
    }
    if "catboost" in sanitized:
        catboost = sanitized["catboost"]
        bootstrap_type = str(catboost.get("bootstrap_type", "Bayesian")).lower()
        if bootstrap_type == "bayesian":
            catboost.pop("subsample", None)
        catboost.setdefault("logging_level", "Silent")

    if "logreg" in sanitized:
        logreg = sanitized["logreg"]
        penalty = logreg.pop("penalty", None)
        if penalty == "l1":
            logreg["l1_ratio"] = 1.0
        elif penalty == "l2":
            logreg["l1_ratio"] = 0.0
        elif penalty == "none":
            logreg["C"] = 1e12
            logreg["l1_ratio"] = 0.0
        elif penalty == "elasticnet":
            logreg.setdefault("l1_ratio", 0.5)
        logreg.setdefault("max_iter", 50000)

    if "xgb" in sanitized:
        xgb_params = sanitized["xgb"]
        xgb_params.setdefault("objective", "binary:logistic")
        xgb_params.setdefault("eval_metric", "logloss")

    return sanitized


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


def _normalize_model_names(include: Sequence[str] | None = None) -> tuple[str, ...]:
    known_models = tuple(DEFAULT_MODEL_PARAMS)
    names = tuple(include or known_models)
    unknown = sorted(set(names) - set(known_models))
    if unknown:
        known = ", ".join(known_models)
        raise ValueError(f"Unknown candidate models: {unknown}. Known models: {known}")
    return names


def _take_rows(data: Any, indices: Any) -> Any:
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def _summarize_fold_scores(
    fold_scores: Mapping[str, Sequence[Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    import numpy as np

    summary: dict[str, dict[str, float]] = {}
    metric_names = ("roc_auc", "accuracy", "f1")
    for model_name, scores in fold_scores.items():
        summary[model_name] = {}
        for metric in metric_names:
            values = np.asarray([score[metric] for score in scores], dtype=float)
            summary[model_name][metric] = float(values.mean())
            summary[model_name][f"{metric}_std"] = float(values.std(ddof=0))
    return summary


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
