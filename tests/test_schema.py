import unittest

from spotify_hit_model.schema import (
    IGNORED_MODEL_FEATURES,
    MODEL_INPUT_COLUMNS,
    normalize_decade,
    prepare_model_records,
)


class SchemaTests(unittest.TestCase):
    def test_public_ignored_fields_are_removed_from_model_records(self):
        record = {
            "artist": "The Beatles",
            "track": "Hey Jude",
            "decade_of_release": 1960,
            "danceability": 0.5,
            "energy": 0.7,
            "key": 7,
            "loudness": -8.5,
            "mode": 1,
            "speechiness": 0.03,
            "acousticness": 0.2,
            "instrumentalness": 0.0,
            "liveness": 0.1,
            "valence": 0.8,
            "tempo": 120.0,
            "duration_ms": 431000,
            "time_signature": 4,
            "chorus_hit": 0.5,
            "sections": 8,
        }

        prepared = prepare_model_records([record])

        self.assertEqual(len(prepared), 1)
        self.assertEqual(set(IGNORED_MODEL_FEATURES), {"speechiness", "instrumentalness"})
        self.assertNotIn("speechiness", prepared[0])
        self.assertNotIn("instrumentalness", prepared[0])
        self.assertEqual(set(prepared[0]), set(MODEL_INPUT_COLUMNS))
        self.assertEqual(prepared[0]["decade_of_release"], "60")

    def test_ignored_fields_do_not_affect_prepared_model_record(self):
        base = {
            "artist": "Queen",
            "track": "Bohemian Rhapsody",
            "decade_of_release": 1970,
            "danceability": 0.3,
            "energy": 0.6,
            "key": 0,
            "loudness": -7.2,
            "mode": 1,
            "speechiness": 0.05,
            "acousticness": 0.1,
            "instrumentalness": 0.0,
            "liveness": 0.2,
            "valence": 0.4,
            "tempo": 72.0,
            "duration_ms": 355000,
            "time_signature": 4,
            "chorus_hit": 0.3,
            "sections": 12,
        }
        changed = dict(base, speechiness=0.99, instrumentalness=0.88)

        self.assertEqual(prepare_model_records([base]), prepare_model_records([changed]))

    def test_unknown_public_fields_are_rejected(self):
        record = {
            "artist": "The Beatles",
            "track": "Hey Jude",
            "decade_of_release": 1960,
            "danceability": 0.5,
            "energy": 0.7,
            "key": 7,
            "loudness": -8.5,
            "mode": 1,
            "speechiness": 0.03,
            "acousticness": 0.2,
            "instrumentalness": 0.0,
            "liveness": 0.1,
            "valence": 0.8,
            "tempo": 120.0,
            "duration_ms": 431000,
            "time_signature": 4,
            "chorus_hit": 0.5,
            "sections": 8,
            "surprise": 1,
        }

        with self.assertRaisesRegex(ValueError, "Unknown input fields"):
            prepare_model_records([record])

    def test_normalize_decade(self):
        self.assertEqual(normalize_decade(1964), "60")
        self.assertEqual(normalize_decade("1980"), "80")
        self.assertEqual(normalize_decade(2007), "0")
        self.assertEqual(normalize_decade("10"), "10")
        self.assertEqual(normalize_decade(1955), "unknown")


if __name__ == "__main__":
    unittest.main()
