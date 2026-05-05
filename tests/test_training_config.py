import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd

import spotify_hit_model.training as training
from spotify_hit_model.tabm_model import TabMClassifier


class FakeCatBoostClassifier:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return [[0.5, 0.5] for _ in range(len(X))]


class TrainingConfigTests(unittest.TestCase):
    def test_catboost_candidate_does_not_pass_subsample(self):
        previous_catboost = sys.modules.get("catboost")
        previous_preprocessor = training.create_preprocessor
        sys.modules["catboost"] = types.SimpleNamespace(CatBoostClassifier=FakeCatBoostClassifier)
        training.create_preprocessor = lambda X: "passthrough"
        self.addCleanup(self._restore, previous_catboost, previous_preprocessor)

        X = pd.DataFrame({"danceability": [0.1, 0.2, 0.3, 0.4]})
        y = [0, 1, 0, 1]

        training.fit_candidate_models(X, y, include=("catboost",), tree_device="cpu")

        self.assertNotIn("subsample", FakeCatBoostClassifier.last_kwargs)

    def test_catboost_sanitizer_keeps_subsample_only_for_compatible_bootstrap(self):
        bayesian = training.sanitize_model_params(
            {"catboost": {"depth": 7, "subsample": 0.5}}
        )
        bernoulli = training.sanitize_model_params(
            {"catboost": {"bootstrap_type": "Bernoulli", "subsample": 0.5}}
        )

        self.assertNotIn("subsample", bayesian["catboost"])
        self.assertEqual(bernoulli["catboost"]["subsample"], 0.5)

    def test_catboost_gpu_sanitizer_removes_unsupported_rsm_options(self):
        sanitized = training.sanitize_model_params(
            {
                "catboost": {
                    "task_type": "GPU",
                    "colsample_bylevel": 0.4,
                    "rsm": 0.5,
                }
            }
        )

        self.assertNotIn("colsample_bylevel", sanitized["catboost"])
        self.assertNotIn("rsm", sanitized["catboost"])

    def test_catboost_factory_removes_rsm_after_gpu_device_params_are_merged(self):
        previous_catboost = sys.modules.get("catboost")
        sys.modules["catboost"] = types.SimpleNamespace(CatBoostClassifier=FakeCatBoostClassifier)
        self.addCleanup(self._restore_catboost_module, previous_catboost)

        factory = training.build_model_factories(
            include=("catboost",),
            model_params={"catboost": {"colsample_bylevel": 0.4}},
            device_params={"catboost": {"task_type": "GPU"}},
        )["catboost"]
        factory()

        self.assertEqual(FakeCatBoostClassifier.last_kwargs["task_type"], "GPU")
        self.assertNotIn("colsample_bylevel", FakeCatBoostClassifier.last_kwargs)

    def test_model_params_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_params.json"
            training.save_model_params(
                {
                    "catboost": {"depth": 4, "subsample": 0.8},
                    "logreg": {"penalty": "none", "C": 10.0},
                },
                path,
            )

            loaded = training.load_model_params(path)

        self.assertEqual(loaded["catboost"]["depth"], 4)
        self.assertNotIn("subsample", loaded["catboost"])
        self.assertNotIn("penalty", loaded["logreg"])
        self.assertEqual(loaded["logreg"]["C"], 1e12)
        self.assertEqual(loaded["logreg"]["l1_ratio"], 0.0)

    def test_select_ensemble_from_oof_keeps_single_model_when_ensemble_does_not_improve(self):
        y = [0, 0, 1, 1]
        selection = training.select_ensemble_from_oof(
            {
                "tabm": [0.1, 0.2, 0.8, 0.9],
                "xgb": [0.9, 0.8, 0.2, 0.1],
            },
            y,
            min_improvement=0.001,
        )

        self.assertEqual(selection.selected_weights, {"tabm": 1.0})
        self.assertEqual(selection.ensemble_score, 1.0)

    def test_tabm_defaults_to_early_stopping(self):
        model = TabMClassifier()

        self.assertTrue(model.early_stopping)
        self.assertEqual(model.patience, 10)
        self.assertGreater(model.epochs, model.patience)

    @staticmethod
    def _restore(previous_catboost, previous_preprocessor):
        if previous_catboost is None:
            sys.modules.pop("catboost", None)
        else:
            sys.modules["catboost"] = previous_catboost
        training.create_preprocessor = previous_preprocessor

    @staticmethod
    def _restore_catboost_module(previous_catboost):
        if previous_catboost is None:
            sys.modules.pop("catboost", None)
        else:
            sys.modules["catboost"] = previous_catboost


if __name__ == "__main__":
    unittest.main()
