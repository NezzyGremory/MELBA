"""Public MELBA business-data regression estimator."""

from __future__ import annotations

import numpy as np

from .core.optimization import optimize_huber
from .core.validation import (
    as_features,
    as_target,
    validate_epochs,
    validate_positive_real,
    validate_random_state,
    validate_training_data,
)
from .evaluation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from .utils.preprocessing import BusinessFeatureMap


class MELBA:
    """Fit a regularized Huber regression model to business data.

    By default, the model adds squared and pairwise feature terms to capture
    simple curvature and interactions. This is a classical quadratic feature
    regression formulation, not a claim of a new learning method.
    """

    def __init__(
        self,
        learning_rate: float = 0.5,
        epochs: int = 1000,
        tolerance: float = 1e-8,
        random_state: int | None = 42,
        l2_penalty: float = 0.01,
        huber_delta: float = 1.345,
        quadratic_terms: bool = True,
    ) -> None:
        self.learning_rate = validate_positive_real(learning_rate, "learning_rate")
        if self.learning_rate > 1.0:
            raise ValueError("learning_rate must be greater than 0 and at most 1.")
        self.epochs = validate_epochs(epochs)
        self.tolerance = validate_positive_real(tolerance, "tolerance", allow_zero=True)
        self.random_state = validate_random_state(random_state)
        self.l2_penalty = validate_positive_real(l2_penalty, "l2_penalty", allow_zero=True)
        self.huber_delta = validate_positive_real(huber_delta, "huber_delta")
        if not isinstance(quadratic_terms, (bool, np.bool_)):
            raise ValueError("quadratic_terms must be a boolean.")
        self.quadratic_terms = bool(quadratic_terms)

        self.is_fitted_ = False
        self.n_features_in_: int | None = None
        self.feature_map_: BusinessFeatureMap | None = None
        self.weights_: np.ndarray | None = None
        self.target_mean_: float | None = None
        self.target_scale_: float | None = None
        self.loss_history_: list[float] = []
        self.loss_: float | None = None
        self.n_iter_: int = 0
        self.converged_: bool = False

    def fit(self, X: object, y: object) -> "MELBA":
        """Validate data and optimize model parameters, then return ``self``."""
        features, target = validate_training_data(X, y)
        feature_map = BusinessFeatureMap(self.quadratic_terms)
        mapped = feature_map.fit_transform(features)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            target_mean = float(np.mean(target))
            target_scale = float(np.std(target))
        if not np.isfinite(target_mean) or not np.isfinite(target_scale):
            raise ValueError("y values are too large to normalize safely.")
        if target_scale <= np.finfo(float).eps:
            target_scale = 1.0
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            normalized_target = (target - target_mean) / target_scale
        if not np.isfinite(normalized_target).all():
            raise ValueError("y values are too large to normalize safely.")

        weights, history, converged = optimize_huber(
            mapped,
            normalized_target,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            tolerance=self.tolerance,
            l2_penalty=self.l2_penalty,
            huber_delta=self.huber_delta,
            random_state=self.random_state,
        )

        self.feature_map_ = feature_map
        self.weights_ = weights
        self.target_mean_ = target_mean
        self.target_scale_ = target_scale
        self.n_features_in_ = features.shape[1]
        self.loss_history_ = history
        self.loss_ = history[-1]
        self.n_iter_ = len(history) - 1
        self.converged_ = converged
        self.is_fitted_ = True
        return self

    def predict(self, X: object) -> np.ndarray:
        """Predict target values for one or more feature rows."""
        if not self.is_fitted_ or self.feature_map_ is None or self.weights_ is None:
            raise RuntimeError("MELBA must be fitted before calling predict.")
        features = as_features(X)
        mapped = self.feature_map_.transform(features)
        design = np.column_stack((np.ones(features.shape[0]), mapped))
        assert self.target_mean_ is not None and self.target_scale_ is not None
        with np.errstate(over="ignore", invalid="ignore"):
            predictions = design @ self.weights_ * self.target_scale_ + self.target_mean_
        if not np.isfinite(predictions).all():
            raise ValueError("Prediction produced nonfinite values; check the input scale.")
        return predictions

    def evaluate(self, X: object, y: object) -> dict[str, float]:
        """Return MAE, MSE, RMSE, and R-squared on the supplied data."""
        actual = as_target(y)
        predictions = self.predict(X)
        if actual.size != predictions.size:
            raise ValueError(
                "X and y must contain the same number of samples; "
                f"got {predictions.size} and {actual.size}."
            )
        return {
            "mae": mean_absolute_error(actual, predictions),
            "mse": mean_squared_error(actual, predictions),
            "rmse": root_mean_squared_error(actual, predictions),
            "r2": r2_score(actual, predictions),
        }

