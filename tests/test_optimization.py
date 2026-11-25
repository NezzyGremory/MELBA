import numpy as np

from melba.core.learning import huber_gradient, huber_loss
from melba.core.optimization import optimize_huber


def _objective(augmented, target, weights, delta, l2_penalty):
    residuals = augmented @ weights - target
    return float(
        np.mean(huber_loss(residuals, delta))
        + 0.5 * l2_penalty * np.sum(weights[1:] ** 2)
    )


def _central_difference(objective, weights, epsilon=1e-6):
    gradient = np.empty_like(weights)
    for index in range(weights.size):
        positive = weights.copy()
        negative = weights.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        gradient[index] = (objective(positive) - objective(negative)) / (2 * epsilon)
    return gradient


def test_huber_and_l2_gradient_matches_finite_difference_and_optimizer_step():
    design = np.array(
        [[0.2, -0.5], [1.0, 1.0], [-1.0, 0.3], [0.5, -0.8], [1.2, 0.4]]
    )
    target = np.array([0.05, 1.9, -1.7, 2.2, -0.2])
    augmented = np.column_stack((np.ones(target.size), design))
    delta = 0.7
    l2_penalty = 0.13
    weights = np.array([0.15, -0.22, 0.41])

    residuals = augmented @ weights - target
    assert np.any(np.abs(residuals) < delta)
    assert np.any(np.abs(residuals) > delta)
    analytic = augmented.T @ huber_gradient(residuals, delta) / target.size
    analytic[1:] += l2_penalty * weights[1:]
    objective = lambda candidate: _objective(
        augmented, target, candidate, delta, l2_penalty
    )
    numeric = _central_difference(objective, weights)

    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-7)

    seed = 29
    learning_rate = 0.4
    initial_weights = np.zeros(design.shape[1] + 1)
    initial_weights[1:] = np.random.default_rng(seed).normal(
        0.0, 0.01, size=design.shape[1]
    )
    initial_numeric_gradient = _central_difference(
        lambda candidate: _objective(
            augmented, target, candidate, delta, l2_penalty
        ),
        initial_weights,
    )
    lipschitz_bound = np.sum(augmented * augmented) / target.size + l2_penalty
    expected_weights = initial_weights - (
        learning_rate / max(lipschitz_bound, np.finfo(float).eps)
    ) * initial_numeric_gradient

    actual_weights, history, _ = optimize_huber(
        design,
        target,
        learning_rate=learning_rate,
        epochs=1,
        tolerance=0.0,
        l2_penalty=l2_penalty,
        huber_delta=delta,
        random_state=seed,
    )

    np.testing.assert_allclose(actual_weights, expected_weights, rtol=2e-6, atol=2e-8)
    assert len(history) == 2
