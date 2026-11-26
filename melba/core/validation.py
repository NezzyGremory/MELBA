"""Input checks shared by the model and evaluation metrics."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def _real_array(values: object, name: str) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty numeric array.") from exc

    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values.")
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    return array


def as_features(values: object, name: str = "X") -> np.ndarray:
    """Return features as a finite, two-dimensional float array."""
    array = _real_array(values, name)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one feature.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or infinity.")
    return array


def as_target(values: object, name: str = "y") -> np.ndarray:
    """Return a finite, one-dimensional target array."""
    array = _real_array(values, name)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional or a single-column array.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or infinity.")
    return array


def validate_training_data(X: object, y: object) -> tuple[np.ndarray, np.ndarray]:
    """Validate a feature matrix and its matching target."""
    features = as_features(X)
    target = as_target(y)
    if features.shape[0] != target.size:
        raise ValueError(
            "X and y must contain the same number of samples; "
            f"got {features.shape[0]} and {target.size}."
        )
    return features, target


def validate_positive_real(value: object, name: str, *, allow_zero: bool = False) -> float:
    """Validate and normalize a finite positive real parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    normalized = float(value)
    minimum_ok = normalized >= 0 if allow_zero else normalized > 0
    if not np.isfinite(normalized) or not minimum_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return normalized


def validate_epochs(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("epochs must be a positive integer.")
    return int(value)


def validate_random_state(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("random_state must be a non-negative integer or None.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError("random_state must be a non-negative integer or None.")
    return normalized

