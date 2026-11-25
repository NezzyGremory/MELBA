import numpy as np
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


def test_metrics_remain_finite_for_extreme_inputs_when_result_is_representable():
    actual = [1e200, 2e200]
    predicted = [0.0, 0.0]

    assert mean_absolute_error(actual, predicted) == pytest.approx(1.5e200)
    assert root_mean_squared_error(actual, predicted) == pytest.approx(1.5811388300841898e200)
    assert r2_score(actual, predicted) == pytest.approx(-9.0)
    with pytest.raises(OverflowError, match="MSE.*finite float64"):
        mean_squared_error(actual, predicted)

    mse = mean_squared_error([1e150, 2e150], [0.0, 0.0])
    assert mse == pytest.approx(2.5e300)
    assert np.isfinite(mse)

    mae = mean_absolute_error([1.7e308, 0.0, 0.0, 0.0], [-1.7e308, 0.0, 0.0, 0.0])
    assert mae == pytest.approx(8.5e307)


def test_metrics_raise_when_extreme_result_is_not_representable():
    with pytest.raises(OverflowError, match="MAE.*finite float64"):
        mean_absolute_error([1.7e308, -1.7e308], [-1.7e308, 1.7e308])
    with pytest.raises(OverflowError, match="RMSE.*finite float64"):
        root_mean_squared_error([1.7e308], [-1.7e308])
    with pytest.raises(OverflowError, match="R-squared.*finite float64"):
        r2_score([0.0, 1.0], [1e308, -1e308])


def test_constant_target_r2_behavior_is_preserved_for_extreme_predictions():
    assert r2_score([4.0, 4.0], [1e308, -1e308]) == 0.0


def test_evaluate_returns_all_metrics():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [1.0, 2.0, 3.0, 4.0]
    model = MELBA(quadratic_terms=False, epochs=300).fit(X, y)

    result = model.evaluate([[4.0], [5.0]], [5.0, 6.0])

    assert set(result) == {"mae", "mse", "rmse", "r2"}
    assert result["mae"] >= 0
    assert result["mse"] >= 0
    assert result["rmse"] == pytest.approx(result["mse"] ** 0.5)


def test_evaluate_computes_metrics_in_original_target_units():
    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = np.array([12000.0, 12780.0, 13650.0, 14120.0, 15100.0, 15750.0])
    model = MELBA(quadratic_terms=False, epochs=250, random_state=8).fit(X, y)
    predictions = model.predict(X)

    result = model.evaluate(X, y)

    residuals = y - predictions
    assert result["mae"] == pytest.approx(np.mean(np.abs(residuals)))
    assert result["mse"] == pytest.approx(np.mean(residuals**2))
    assert result["rmse"] == pytest.approx(np.sqrt(np.mean(residuals**2)))
    assert result["r2"] == pytest.approx(
        1.0 - np.sum(residuals**2) / np.sum((y - np.mean(y)) ** 2)
    )


def test_evaluate_rejects_a_target_sample_count_mismatch():
    model = MELBA().fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="same number of samples"):
        model.evaluate([[4.0], [5.0]], [4.0])
