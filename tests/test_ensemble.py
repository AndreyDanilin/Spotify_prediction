import unittest

from spotify_hit_model.ensemble import WeightedSoftVotingEnsemble, select_weighted_models


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.seen = None

    def predict_proba(self, records):
        self.seen = records
        return self.probabilities


class WeightedSoftVotingEnsembleTests(unittest.TestCase):
    def test_predict_proba_returns_weighted_average(self):
        ensemble = WeightedSoftVotingEnsemble(
            models={
                "a": FakeModel([[0.2, 0.8], [0.7, 0.3]]),
                "b": FakeModel([[0.6, 0.4], [0.1, 0.9]]),
            },
            weights={"a": 0.75, "b": 0.25},
        )

        probabilities = ensemble.predict_proba([{"x": 1}, {"x": 2}])

        self.assertEqual(probabilities, [[0.3, 0.7], [0.55, 0.45]])

    def test_predict_uses_positive_class_threshold(self):
        ensemble = WeightedSoftVotingEnsemble(
            models={
                "a": FakeModel([[0.2, 0.8], [0.7, 0.3]]),
                "b": FakeModel([[0.6, 0.4], [0.1, 0.9]]),
            },
            weights={"a": 0.75, "b": 0.25},
        )

        self.assertEqual(ensemble.predict([{"x": 1}, {"x": 2}]), [1, 0])

    def test_select_weighted_models_keeps_best_models_and_normalizes_weights(self):
        selected = select_weighted_models(
            {"xgb": 0.962, "catboost": 0.955, "tabm": 0.91, "logreg": 0.88},
            tolerance=0.01,
            min_models=2,
        )

        self.assertEqual(set(selected), {"xgb", "catboost"})
        self.assertAlmostEqual(sum(selected.values()), 1.0)
        self.assertGreater(selected["xgb"], selected["catboost"])

    def test_ensemble_rejects_missing_weight(self):
        with self.assertRaisesRegex(ValueError, "Missing weights"):
            WeightedSoftVotingEnsemble(models={"a": FakeModel([[0.4, 0.6]])}, weights={})


if __name__ == "__main__":
    unittest.main()
