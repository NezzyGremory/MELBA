"""Huber objective and derivative used by MELBA's trainer."""

from __future__ import annotations

import numpy as np


def huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Compute elementwise Huber loss for residuals and threshold ``delta``."""
    absolute = np.abs(residuals)
    return np.where(
        absolute <= delta,
        0.5 * residuals**2,
        delta * (absolute - 0.5 * delta),
    )


def huber_gradient(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Compute the derivative of Huber loss with respect to each residual."""
    return np.clip(residuals, -delta, delta)

