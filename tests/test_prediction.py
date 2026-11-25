import numpy as np
import pandas as pd
import pytest

from melba import MELBA


def fitted_model():
    X = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0], [5.0, 4.0]])
    y = np.array([8.0, 7.0, 8.0, 11.0, 16.0])
    return MELBA(epochs=250, random_state=3).fit(X, y)


def test_prediction_returns_one_value_per_row():
    model = fitted_model()
    predictions = model.predict([[2.5, 2.5], [3.5, 2.0]])

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()


def test_pandas_dataframes_are_accepted():
    X = pd.DataFrame({"price": [1.0, 2.0, 3.0], "promotion": [0.0, 2.0, 1.0]})
    y = pd.Series([5.0, 7.0, 8.0])
    model = MELBA(epochs=100, random_state=5).fit(X, y)

    assert model.predict(X).shape == (3,)


def test_prediction_before_training_raises():
    with pytest.raises(RuntimeError, match="fitted before calling predict"):
        MELBA().predict([[1.0]])


def test_prediction_rejects_a_different_feature_count():
    with pytest.raises(ValueError, match="different number of features"):
        fitted_model().predict([[1.0, 2.0, 3.0]])


def test_prediction_rejects_values_that_overflow_feature_scaling():
    with pytest.raises(ValueError, match="too large to transform safely"):
        fitted_model().predict([[1e308, 1e308]])


@pytest.mark.parametrize(
    "X",
    [[], [[1.0, float("nan")]], [[float("inf"), 1.0]], [["low", 2.0]], [[[1.0, 2.0]]]],
)
def test_prediction_rejects_empty_nonfinite_nonnumeric_and_bad_dimensions(X):
    with pytest.raises(ValueError):
        fitted_model().predict(X)


def test_same_seed_produces_same_fit_and_predictions():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = 3.0 * X[:, 0] - 0.5 * X[:, 1] + np.sin(X[:, 0])
    first = MELBA(random_state=81, epochs=350).fit(X, y)
    second = MELBA(random_state=81, epochs=350).fit(X, y)

    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert first.loss_history_ == second.loss_history_
