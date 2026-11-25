"""Numerically stable regression metrics with explicit constant-target handling."""

from __future__ import annotations

import math

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


def _scaled_residuals(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, np.ndarray]:
    """Return residuals scaled to avoid overflow in subtraction and squaring."""
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = actual - predicted
    if np.isfinite(residuals).all():
        scale = float(np.max(np.abs(residuals)))
        if scale == 0.0:
            return 0.0, np.zeros_like(actual)
        return scale, residuals / scale

    scale = max(float(np.max(np.abs(actual))), float(np.max(np.abs(predicted))))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized_residuals = actual / scale - predicted / scale
    return scale, normalized_residuals


def _checked_scaled_product(scale: float, factor: float, metric: str) -> float:
    """Scale a finite nonnegative factor, raising if the result overflows float64."""
    if scale == 0.0 or factor == 0.0:
        return 0.0
    maximum = float(np.finfo(float).max)
    if factor > 1.0 and scale > maximum / factor:
        raise OverflowError(f"{metric} is outside the finite float64 range.")
    result = scale * factor
    if not math.isfinite(result):
        raise OverflowError(f"{metric} is outside the finite float64 range.")
    return float(result)


def mean_absolute_error(y_true: object, y_pred: object) -> float:
    """Return mean absolute error without overflowing during subtraction."""
    actual, predicted = _matching_targets(y_true, y_pred)
    scale, residuals = _scaled_residuals(actual, predicted)
    normalized_error = float(np.mean(np.abs(residuals)))
    return _checked_scaled_product(scale, normalized_error, "MAE")


def _root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    scale, residuals = _scaled_residuals(actual, predicted)
    normalized_mse = float(np.mean(residuals * residuals))
    return _checked_scaled_product(scale, math.sqrt(normalized_mse), "RMSE")


def mean_squared_error(y_true: object, y_pred: object) -> float:
    """Return MSE, raising if its final value cannot fit in a float64."""
    actual, predicted = _matching_targets(y_true, y_pred)
    scale, residuals = _scaled_residuals(actual, predicted)
    normalized_mse = float(np.mean(residuals * residuals))
    if scale == 0.0 or normalized_mse == 0.0:
        return 0.0

    scale_mantissa, scale_exponent = math.frexp(scale)
    mse_mantissa, mse_exponent = math.frexp(normalized_mse)
    try:
        mse = math.ldexp(
            scale_mantissa * scale_mantissa * mse_mantissa,
            2 * scale_exponent + mse_exponent,
        )
    except OverflowError as exc:
        raise OverflowError("MSE is outside the finite float64 range.") from exc
    if not math.isfinite(mse):
        raise OverflowError("MSE is outside the finite float64 range.")
    return float(mse)


def root_mean_squared_error(y_true: object, y_pred: object) -> float:
    """Return RMSE directly, without first computing a possibly overflowing MSE."""
    actual, predicted = _matching_targets(y_true, y_pred)
    return _root_mean_squared_error(actual, predicted)


def r2_score(y_true: object, y_pred: object) -> float:
    """Return R-squared using scaled sums of squares.

    Raises ``OverflowError`` when a finite input pair implies an R-squared below
    the range representable by a finite float64.
    """
    actual, predicted = _matching_targets(y_true, y_pred)
    if np.all(actual == actual[0]):
        return 1.0 if np.array_equal(actual, predicted) else 0.0

    residual_scale, residuals = _scaled_residuals(actual, predicted)
    residual_sum_scaled = float(np.sum(residuals * residuals))
    if residual_sum_scaled == 0.0:
        return 1.0

    with np.errstate(over="ignore", invalid="ignore"):
        target_differences = actual - actual[0]
    if np.isfinite(target_differences).all():
        target_scale = float(np.max(np.abs(target_differences)))
        scaled_target = target_differences / target_scale
    else:
        target_scale = float(np.max(np.abs(actual)))
        scaled_target = actual / target_scale - actual[0] / target_scale
    centered_target = scaled_target - np.mean(scaled_target)
    total_sum_scaled = float(np.sum(centered_target * centered_target))
    if total_sum_scaled == 0.0:
        raise ArithmeticError("Could not resolve target variance at float64 precision.")

    log_ratio = (
        2.0 * (math.log(residual_scale) - math.log(target_scale))
        + math.log(residual_sum_scaled)
        - math.log(total_sum_scaled)
    )
    max_log = math.log(float(np.finfo(float).max))
    if log_ratio > max_log:
        raise OverflowError("R-squared is outside the finite float64 range.")
    min_log = math.log(float(np.nextafter(0.0, 1.0)))
    ratio = 0.0 if log_ratio < min_log else math.exp(log_ratio)
    score = 1.0 - ratio
    if not math.isfinite(score):
        raise OverflowError("R-squared is outside the finite float64 range.")
    return float(score)
