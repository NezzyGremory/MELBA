# MELBA framework

## Scope

MELBA 0.2.0 is a Python library for numeric business-data regression and formula-based economic mathematics. It supports model fitting, prediction, and evaluation in notebooks or Python scripts, with a separate economic functions package. It has no server or database component.

## Package layout

- `melba/model.py` implements the public `MELBA` estimator.
- `melba/core/validation.py` checks feature, target, and configuration values.
- `melba/core/learning.py` implements the Huber loss and its residual gradient.
- `melba/core/optimization.py` runs full-batch gradient descent.
- `melba/utils/preprocessing.py` fits and applies feature scaling and quadratic expansion.
- `melba/evaluation/metrics.py` implements MAE, MSE, RMSE, and R-squared.
- `melba/economics/` implements linear economic functions, polynomial derivatives, bounded profit optimization, and price elasticity. These formulas are independent of the MELBA estimator and optimizer.
- `examples/` contains runnable synthetic examples.
- `tests/` checks estimator behavior, input validation, and metrics.

## Estimator contract

```python
from melba import MELBA

model = MELBA(
    learning_rate=0.5,
    epochs=1000,
    tolerance=1e-8,
    random_state=42,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
```

`fit` returns the estimator. `predict` returns a one-dimensional NumPy array and raises an error before fitting. `evaluate` returns a dictionary with `mae`, `mse`, `rmse`, and `r2` keys. DataFrames and Series work as inputs through their numeric array representation; MELBA does not align or impute missing data.

`learning_rate` is a multiplier in `(0, 1]` on a step size normalized by a safe Frobenius-norm bound. `epochs` sets the maximum number of full-batch updates. `tolerance` stops training when the absolute change in the objective is at most that value. `random_state` seeds the small initial coefficient values; passing `None` uses a fresh random seed. `l2_penalty`, `huber_delta`, and `quadratic_terms` configure the objective and feature mapping.

## Dependencies and use

The package requires Python 3.10 or newer, NumPy, and Pandas. Matplotlib is an optional dependency used by the notebook. Pytest is an optional development dependency. The core training path does not import scikit-learn.

The project is packaged with `pyproject.toml` and setuptools. Install from a checkout with `python -m pip install -e .`. The notebook includes dependency and source-bootstrap cells intended for Colab and Jupyter; a live Colab session was not available for verification here.

## Known limits

The quadratic expansion creates O(p²) terms for p input columns, so this release is intended for modest feature counts. Inputs must already be numeric and complete. The model makes no causal claim and offers no built-in feature selection, cross-validation, or uncertainty estimates. Evaluation scores describe only the supplied sample.

The economics functions calculate only the relationships and parameters supplied by the caller. Profit optimization is bounded by a finite quantity interval, and elasticity describes a mathematical function rather than a causal effect. See [Economic Mathematics](economics.md) for formulas, API details, assumptions, and examples.
