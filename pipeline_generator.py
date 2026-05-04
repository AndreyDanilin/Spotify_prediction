from __future__ import annotations

import argparse
import os
from pathlib import Path

from spotify_hit_model.training import train_and_save_ensemble


DEFAULT_ARTIFACT_PATH = Path("music-classifier/app/hit_ensemble.joblib")
DEFAULT_FEATURE_CACHE_PATH = Path("artifacts/training_features.joblib")


def main() -> None:
    args = parse_args()
    configure_environment(args)
    run = train_and_save_ensemble(
        data_dir=args.data_dir,
        artifact_path=args.artifact_path,
        embedding_model_name=args.embedding_model_name,
        feature_cache_path=args.feature_cache_path,
        rebuild_feature_cache=args.rebuild_feature_cache,
        candidate_models=args.models,
        model_params_path=args.model_params_path,
        cv_splits=args.cv_splits,
        ensemble_min_improvement=args.ensemble_min_improvement,
        tabm_epochs=args.tabm_epochs,
        tabm_device=args.tabm_device,
        tree_device=args.tree_device,
        verbose=True,
    )
    print(f"Ensemble artifact saved to {run.artifact_path}")
    print("Cross-validation scores:")
    for name, scores in run.validation_scores.items():
        print(
            f"  {name}: "
            f"roc_auc={scores['roc_auc']:.4f}, "
            f"roc_auc_std={scores.get('roc_auc_std', 0.0):.4f}, "
            f"accuracy={scores['accuracy']:.4f}, "
            f"f1={scores['f1']:.4f}"
        )
    print("Selected weights:")
    for name, weight in run.selected_weights.items():
        print(f"  {name}: {weight:.4f}")
    print("Test scores:")
    for name, scores in run.test_scores.items():
        print(
            f"  {name}: "
            f"roc_auc={scores['roc_auc']:.4f}, "
            f"accuracy={scores['accuracy']:.4f}, "
            f"f1={scores['f1']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save the Spotify hit weighted ensemble artifact."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--embedding-model-name", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--feature-cache-path",
        type=Path,
        default=DEFAULT_FEATURE_CACHE_PATH,
        help="Cache file for generated track embeddings and model features.",
    )
    parser.add_argument(
        "--rebuild-feature-cache",
        action="store_true",
        help="Recompute embeddings even if the feature cache exists.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=("xgb", "catboost", "logreg", "tabm"),
        metavar="MODEL",
        help="Candidate models to train: xgb catboost logreg tabm.",
    )
    parser.add_argument(
        "--model-params-path",
        type=Path,
        default=Path("artifacts/model_params.json"),
        help="JSON file with Optuna best parameters to reuse for final training.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Stratified CV folds for model and ensemble selection.",
    )
    parser.add_argument(
        "--ensemble-min-improvement",
        type=float,
        default=0.001,
        help="Minimum OOF ROC-AUC gain required to add another model to the ensemble.",
    )
    parser.add_argument("--tabm-epochs", type=int, default=80)
    parser.add_argument(
        "--tabm-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="TabM device. Use cuda for a hard CUDA requirement.",
    )
    parser.add_argument(
        "--tree-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="XGBoost/CatBoost device. Use cuda for a hard CUDA requirement.",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--offline-embeddings",
        action="store_true",
        help="Use only locally cached Hugging Face embedding files.",
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    if args.hf_home is not None:
        os.environ["HF_HOME"] = str(args.hf_home)
    if args.offline_embeddings:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


if __name__ == "__main__":
    main()
