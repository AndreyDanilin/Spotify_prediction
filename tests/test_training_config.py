import sys
import types
import unittest

import pandas as pd

import spotify_hit_model.training as training


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

    @staticmethod
    def _restore(previous_catboost, previous_preprocessor):
        if previous_catboost is None:
            sys.modules.pop("catboost", None)
        else:
            sys.modules["catboost"] = previous_catboost
        training.create_preprocessor = previous_preprocessor


if __name__ == "__main__":
    unittest.main()
