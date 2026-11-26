# MELBA

MELBA (Mathematical Economics Learning-Based Algorithm) is an independent Python library for regression on business data and formula-based economic mathematics. Version 0.1.0 provides a small NumPy-based estimator and a separate economics toolkit.

## Purpose

MELBA provides an inspectable starting point for studying relationships among business variables such as price, promotion spend, operating cost, and sales. Its first implementation uses familiar regression methods. It does not claim a new machine-learning method or established scientific novelty.

## Version 0.1.0

- `MELBA.fit(X, y)` validates data and trains the model.
- `MELBA.predict(X)` predicts numeric targets after training.
- `MELBA.evaluate(X, y)` returns MAE, MSE, RMSE, and R-squared.
- Training uses quadratic feature terms by default, Huber loss, L2 regularization, and full-batch gradient descent.
- Inputs may be NumPy arrays or Pandas objects. Features must be finite numeric values.
- A reproducible synthetic sales example and a Colab-ready demonstration notebook are included.

## Install

From a checkout of this project:

```bash
python -m pip install -e .
```

For optional plotting and tests:

```bash
python -m pip install -e ".[plot,test]"
```

## Quick start

```python
import numpy as np
from melba import MELBA

X_train = np.array([[20, 100], [25, 140], [30, 180], [35, 210]], dtype=float)
y_train = np.array([120, 155, 190, 215], dtype=float)
X_test = np.array([[28, 165]], dtype=float)

model = MELBA(learning_rate=0.5, epochs=1000, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
results = model.evaluate(X_test, [175])
print(predictions)
print(results)
```

The example's values are illustrative. They are not company data.

## Examples and tests

Run an end-to-end NumPy example:

```bash
python examples/basic_usage.py
```

Run the synthetic sales example:

```bash
python examples/business_sales.py
```

Run the test suite:

```bash
python -m pytest
```

Open `notebooks/MELBA_demo.ipynb` in Google Colab or Jupyter for a synthetic dataset walkthrough. The notebook contains a source snapshot so it can bootstrap its own MELBA import when opened without a project checkout. The notebook code was exercised locally, but this project has not been run in a live Colab session.

## Project layout

```text
melba/       estimator, economics, validation, optimization, metrics, preprocessing
examples/    regression, synthetic business-sales, and economics examples
tests/       estimator, economics, validation, preprocessing, and metric tests
notebooks/   self-contained demonstration notebook
docs/        architecture and algorithm notes
```

## Algorithm and limitations

MELBA standardizes input columns, adds each standardized feature's square and pairwise products by default, then standardizes the expanded columns. It standardizes the target for optimization, minimizes Huber loss plus an L2 penalty on non-intercept coefficients, and converts predictions back to the target's original scale.

This is a quadratic feature regression model trained with a standard robust loss and gradient descent. The expanded design can grow quadratically with the number of input columns. The model assumes numeric tabular inputs and does not provide categorical encoding, missing-value imputation, feature selection, time-series handling, uncertainty intervals, or causal estimates. R-squared on a constant target is defined as 1 for exact predictions and 0 otherwise.

See the [algorithm documentation](https://github.com/NezzyGremory/MELBA/blob/main/docs/algorithm.md) for the objective and training details. See the [framework documentation](https://github.com/NezzyGremory/MELBA/blob/main/docs/framework.md) for package boundaries and API behavior.

## Economic mathematics

The `melba.economics` package provides linear demand, supply, cost, revenue, and profit functions; analytical polynomial derivatives; bounded profit optimization; and price elasticity calculations. These calculations remain separate from the regression optimizer and require users to supply their own economic parameters and assumptions. See the [Economic Mathematics guide](docs/economics.md) and run `python examples/economic_functions.py` for a complete example.

## Development status

Version 0.1.0 is an initial regression foundation with a separate formula-based economics toolkit. Neither is a validated domain model: assess predictions on held-out data and supply appropriate economic assumptions before using results in business decisions. No claim of algorithmic novelty is made.
