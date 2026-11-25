"""Fit MELBA to a reproducible synthetic sales dataset stored in a DataFrame."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from melba import MELBA


def make_sales_data(sample_count: int = 320, random_state: int = 14) -> pd.DataFrame:
    """Create illustrative data; these values are not observations from a company."""
    rng = np.random.default_rng(random_state)
    price = rng.uniform(12.0, 60.0, sample_count)
    promotion_spend = rng.uniform(0.0, 80.0, sample_count)
    operating_cost = rng.uniform(500.0, 2500.0, sample_count)
    sales = (
        280.0
        - 2.0 * price
        + 1.3 * promotion_spend
        - 0.009 * promotion_spend**2
        - 0.018 * operating_cost
        + 0.008 * price * promotion_spend
        + rng.normal(0.0, 8.0, sample_count)
    )
    return pd.DataFrame(
        {
            "price": price,
            "promotion_spend": promotion_spend,
            "operating_cost": operating_cost,
            "sales": sales,
        }
    )


def main() -> None:
    data = make_sales_data()
    rng = np.random.default_rng(14)
    order = rng.permutation(len(data))
    split = int(0.8 * len(order))
    train = data.iloc[order[:split]]
    test = data.iloc[order[split:]]

    model = MELBA(learning_rate=0.5, epochs=1800, random_state=14)
    model.fit(train[["price", "promotion_spend", "operating_cost"]], train["sales"])
    print("Synthetic sales evaluation:", model.evaluate(
        test[["price", "promotion_spend", "operating_cost"]], test["sales"]
    ))


if __name__ == "__main__":
    main()
