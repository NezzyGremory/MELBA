import pytest

from melba import MELBA
from melba.evaluation import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


def test_metrics_match_known_values():
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 5.0]

    assert mean_absolute_error(actual, predicted) == pytest.approx(2 / 3)
    assert mean_squared_error(actual, predicted) == pytest.approx(4 / 3)
    assert root_mean_squared_error(actual, predicted) == pytest.approx((4 / 3) ** 0.5)
    assert r2_score(actual, predicted) == pytest.approx(-1.0)


def test_r2_for_constant_target_is_one_for_perfect_and_zero_otherwise():
    assert r2_score([4.0, 4.0, 4.0], [4.0, 4.0, 4.0]) == 1.0
    assert r2_score([4.0, 4.0, 4.0], [3.0, 4.0, 5.0]) == 0.0


def test_metrics_reject_mismatched_or_nonfinite_targets():
    with pytest.raises(ValueError, match="same number of samples"):
        mean_absolute_error([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="NaN or infinity"):
        mean_squared_error([1.0, float("nan")], [1.0, 2.0])


def test_evaluate_returns_all_metrics():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [1.0, 2.0, 3.0, 4.0]
    model = MELBA(quadratic_terms=False, epochs=300).fit(X, y)

    result = model.evaluate([[4.0], [5.0]], [5.0, 6.0])

    assert set(result) == {"mae", "mse", "rmse", "r2"}
    assert result["mae"] >= 0
    assert result["mse"] >= 0
    assert result["rmse"] == pytest.approx(result["mse"] ** 0.5)


def test_evaluate_rejects_a_target_sample_count_mismatch():
    model = MELBA().fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="same number of samples"):
        model.evaluate([[4.0], [5.0]], [4.0])
