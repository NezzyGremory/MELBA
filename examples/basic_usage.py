"""Small end-to-end example using NumPy arrays."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from melba import MELBA


def main() -> None:
    rng = np.random.default_rng(21)
    X = rng.uniform(0, 10, size=(120, 2))
    sales = 30 - 1.2 * X[:, 0] + 2.1 * X[:, 1] + 0.15 * X[:, 0] * X[:, 1]
    sales += rng.normal(0, 1.2, size=120)

    order = rng.permutation(X.shape[0])
    split = int(0.8 * len(order))
    train_index, test_index = order[:split], order[split:]

    model = MELBA(learning_rate=0.5, epochs=1200, random_state=21)
    model.fit(X[train_index], sales[train_index])
    predictions = model.predict(X[test_index])
    print("Predictions:", predictions[:5])
    print("Evaluation:", model.evaluate(X[test_index], sales[test_index]))


if __name__ == "__main__":
    main()
