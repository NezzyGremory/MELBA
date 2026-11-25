"""Full-batch gradient descent for the regularized Huber objective."""

from __future__ import annotations

import numpy as np

from .learning import huber_gradient, huber_loss


def optimize_huber(
    design: np.ndarray,
    target: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    tolerance: float,
    l2_penalty: float,
    huber_delta: float,
    random_state: int | None,
) -> tuple[np.ndarray, list[float], bool]:
    """Fit weights using full-batch gradient descent.

    ``design`` excludes the intercept. The intercept is prepended here and is
    left out of the L2 penalty. The Frobenius norm provides a safe upper bound
    for the smooth part's gradient Lipschitz constant.
    """
    sample_count, feature_count = design.shape
    augmented = np.column_stack((np.ones(sample_count), design))

    generator = np.random.default_rng(random_state)
    weights = np.zeros(feature_count + 1)
    weights[1:] = generator.normal(0.0, 0.01, size=feature_count)

    lipschitz_bound = float(np.sum(augmented * augmented) / sample_count + l2_penalty)
    step_size = learning_rate / max(lipschitz_bound, np.finfo(float).eps)

    def objective(current: np.ndarray) -> float:
        residuals = augmented @ current - target
        data_loss = float(np.mean(huber_loss(residuals, huber_delta)))
        penalty = 0.5 * l2_penalty * float(np.sum(current[1:] ** 2))
        return data_loss + penalty

    history = [objective(weights)]
    converged = False
    for _ in range(epochs):
        residuals = augmented @ weights - target
        gradient = augmented.T @ huber_gradient(residuals, huber_delta) / sample_count
        gradient[1:] += l2_penalty * weights[1:]
        weights -= step_size * gradient

        current_loss = objective(weights)
        history.append(current_loss)
        if abs(history[-2] - current_loss) <= tolerance:
            converged = True
            break

    return weights, history, converged

