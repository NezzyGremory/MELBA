import numpy as np
import pytest

from melba import MELBA


def make_data(sample_count=100):
    rng = np.random.default_rng(13)
    X = rng.uniform(-2, 2, size=(sample_count, 2))
    y = 4.0 + 1.7 * X[:, 0] - 0.6 * X[:, 1] + 0.8 * X[:, 0] * X[:, 1]
    y += rng.normal(0, 0.08, size=sample_count)
    return X, y


def test_initialization_exposes_valid_configuration():
    model = MELBA(learning_rate=0.25, epochs=250, random_state=11)

    assert model.learning_rate == 0.25
    assert model.epochs == 250
    assert model.random_state == 11
    assert model.quadratic_terms is True
    assert model.is_fitted_ is False


def test_fit_returns_model_and_reduces_training_loss():
    X, y = make_data()
    model = MELBA(epochs=1200, random_state=9)

    fitted = model.fit(X, y)

    assert fitted is model
    assert model.is_fitted_ is True
    assert model.n_features_in_ == 2
    assert model.n_iter_ > 0
    assert model.loss_ == model.loss_history_[-1]
    assert model.loss_history_[-1] < model.loss_history_[0]


@pytest.mark.parametrize(
    "X,y,message",
    [
        ([], [], "at least one sample"),
        ([[1.0], [2.0]], [1.0], "same number of samples"),
        ([[1.0], [float("nan")]], [1.0, 2.0], "NaN or infinity"),
        ([[1.0], [float("inf")]], [1.0, 2.0], "NaN or infinity"),
        ([["low"], ["high"]], [1.0, 2.0], "real numeric values"),
        ([[[1.0]], [[2.0]]], [1.0, 2.0], "one- or two-dimensional"),
        ([[1.0], [2.0]], [[1.0, 2.0], [3.0, 4.0]], "one-dimensional"),
        ([[1e308], [-1e308]], [1.0, 2.0], "too large to scale safely"),
        ([[1.0], [2.0]], [1e308, -1e308], "too large to normalize safely"),
    ],
)
def test_fit_rejects_invalid_features_and_targets(X, y, message):
    with pytest.raises(ValueError, match=message):
        MELBA().fit(X, y)


def test_fit_accepts_single_column_target():
    model = MELBA(epochs=50).fit([[1.0], [2.0], [3.0]], [[2.0], [4.0], [6.0]])

    assert model.predict([[4.0]]).shape == (1,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0},
        {"learning_rate": 1.01},
        {"learning_rate": float("nan")},
        {"epochs": 0},
        {"epochs": 1.5},
        {"tolerance": -1},
        {"l2_penalty": -0.1},
        {"huber_delta": 0},
        {"random_state": 1.5},
        {"quadratic_terms": "yes"},
    ],
)
def test_invalid_configuration_raises_clear_error(kwargs):
    with pytest.raises(ValueError):
        MELBA(**kwargs)


def test_negative_random_state_is_rejected_during_initialization():
    with pytest.raises(ValueError, match="non-negative integer"):
        MELBA(random_state=-1)
