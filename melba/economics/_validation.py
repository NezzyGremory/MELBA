"""Numeric input checks shared by the economics modules."""

from __future__ import annotations

import numpy as np


def as_finite_real(value: object, name: str, *, nonnegative: bool = False) -> float:
    """Validate one finite real number and return it as a Python float."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number.")
    try:
        result = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number.")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def as_real_values(
    value: object,
    name: str,
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    """Validate a non-empty array of finite real values and return float64."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty numeric value or array.") from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values.")
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
    if nonnegative and np.any(values < 0):
        raise ValueError(f"{name} must contain only non-negative values.")
    return values


def scalar_or_array(values: np.ndarray) -> float | np.ndarray:
    """Return zero-dimensional results as floats and preserve array shapes."""
    if values.ndim == 0:
        return float(values)
    return values


def finite_result(values: np.ndarray, name: str) -> np.ndarray:
    """Raise a useful error if arithmetic produced an unrepresentable result."""
    if not np.isfinite(values).all():
        raise OverflowError(f"{name} calculation produced a non-finite result.")
    return values
