"""Regression evaluation metrics with explicit constant-target handling."""

from __future__ import annotations

import numpy as np

from ..core.validation import as_target


def _matching_targets(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    actual = as_target(y_true, "y_true")
    predicted = as_target(y_pred, "y_pred")
    if actual.size != predicted.size:
        raise ValueError(
            "y_true and y_pred must contain the same number of samples; "
            f"got {actual.size} and {predicted.size}."
        )
    return actual, predicted


def mean_absolute_error(y_true: object, y_pred: object) -> float:
    actual, predicted = _matching_targets(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def mean_squared_error(y_true: object, y_pred: object) -> float:
    actual, predicted = _matching_targets(y_true, y_pred)
    return float(np.mean((actual - predicted) ** 2))


def root_mean_squared_error(y_true: object, y_pred: object) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: object, y_pred: object) -> float:
    actual, predicted = _matching_targets(y_true, y_pred)
    residual_sum = float(np.sum((actual - predicted) ** 2))
    total_sum = float(np.sum((actual - np.mean(actual)) ** 2))
    if total_sum == 0.0:
        return 1.0 if residual_sum == 0.0 else 0.0
    return float(1.0 - residual_sum / total_sum)

