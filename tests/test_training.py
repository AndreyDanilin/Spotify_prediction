import numpy as np
import pytest

from spotify_prediction.training import choose_cv_splits


def test_choose_cv_splits_caps_at_min_class_count() -> None:
    y = np.array([0, 0, 0, 1, 1])

    assert choose_cv_splits(y, requested_splits=5) == 2


def test_choose_cv_splits_rejects_single_class_data() -> None:
    with pytest.raises(ValueError, match="at least two target classes"):
        choose_cv_splits(np.array([1, 1, 1]), requested_splits=5)
