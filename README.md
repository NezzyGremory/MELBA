<p align="center">
  <img src="https://raw.githubusercontent.com/NezzyGremory/MELBA/main/assets/melba-logo.png" alt="MELBA geometric red logo" width="280">
</p>

<h1 align="center">MELBA 0.2.0</h1>

<p align="center">Business data regression and economic mathematics in Python.</p>

<p align="center">
  <a href="https://pypi.org/project/melba/">PyPI project</a> ·
  <a href="https://github.com/NezzyGremory/MELBA">Source repository</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/melba/"><img src="https://img.shields.io/pypi/v/melba" alt="PyPI version"></a>
</p>

MELBA (Mathematical Economics Learning-Based Algorithm) combines a regression engine with a separate economic mathematics toolkit. The 0.2.0 source adds linear economic functions, analytical derivatives, bounded profit optimization, and price elasticity analysis.

> **PyPI status:** Version 0.1.0 is currently published. Version 0.2.0 is not yet available from PyPI. The `pip install melba` command below installs the latest published version; install this 0.2.0 source checkout to use the new economics modules before publication.

## Features

### Regression Engine

- Familiar estimator API: MELBA.fit(X, y), predict(X), and evaluate(X, y).
- Numeric NumPy arrays and Pandas DataFrame/Series inputs.
- Standardized features, optional quadratic and pairwise feature expansion, Huber loss, L2 regularization, and full-batch gradient descent.
- Evaluation metrics: MAE, MSE, RMSE, and R-squared.
- Reproducible initialization through random_state.

The model uses established regression techniques. MELBA makes no claim of algorithmic novelty or scientific superiority.

### Economic Mathematics Engine

Import functions from melba.economics:

- DemandFunction and SupplyFunction for linear demand and supply.
- CostFunction, RevenueFunction, and ProfitFunction for total cost, revenue, and profit.
- PolynomialFunction and analytical derivatives for marginal cost, marginal revenue, and marginal profit.
- Bounded profit optimization that checks stationary candidates and interval boundaries.
- Point price elasticity with elastic, inelastic, or unitary elastic interpretation.

The economic functions use parameters supplied by the caller. They are separate from the regression optimizer and do not infer causal relationships.

## Installation

Install the latest published release from [PyPI](https://pypi.org/project/melba/):

~~~bash
pip install melba
~~~

Or use the Python module form:

~~~bash
python -m pip install --upgrade melba
~~~

To use the 0.2.0 code from a local source checkout:

~~~bash
python -m pip install .
~~~

Matplotlib is optional and only needed for plotting in the notebook. For tests and plotting:

~~~bash
python -m pip install ".[plot,test]"
~~~

## Quick Start

The example below uses synthetic demonstration values. It trains a regression model, predicts sales, and passes the prediction into the economic functions.

~~~python
import numpy as np
from melba import MELBA
from melba.economics import (
    CostFunction,
    DemandFunction,
    Elasticity,
    ProfitFunction,
    RevenueFunction,
    SupplyFunction,
)

# Synthetic demonstration data: promotion budget in thousands -> units sold.
X_train = np.array([[0], [10], [20], [30], [40], [50]], dtype=float)
y_train = np.array([50, 70, 90, 110, 130, 150], dtype=float)

model = MELBA(
    quadratic_terms=False,
    epochs=1000,
    random_state=42,
    l2_penalty=0.0,
)
model.fit(X_train, y_train)

predicted_sales = float(model.predict([[25]])[0])
evaluation = model.evaluate(X_train, y_train)

if predicted_sales < 0:
    raise ValueError("Predicted sales are outside the non-negative quantity domain.")

# Illustrative currency values per product unit.
price_per_unit = 10_000.0
revenue = RevenueFunction().calculate(price_per_unit, predicted_sales)
total_cost = CostFunction(
    fixed_cost=100_000.0,
    variable_cost=6_000.0,
).calculate(predicted_sales)
profit = ProfitFunction().calculate(revenue, total_cost)

# Separate illustrative supply and demand curves.
demand = DemandFunction(a=1_000.0, b=0.05)
supply = SupplyFunction(c=20.0, d=0.02)
demand_quantity = demand.calculate(price_per_unit)
supply_quantity = supply.calculate(price_per_unit)
elasticity = Elasticity.price_demand(demand, price=price_per_unit)

print("Predicted sales:", predicted_sales)
print("Evaluation:", evaluation)
print("Revenue / cost / profit:", revenue, total_cost, profit)
print("Demand / supply:", demand_quantity, supply_quantity)
print("Price elasticity:", elasticity.elasticity, elasticity.category)
~~~

All data and currency values are synthetic and illustrative; they are not company observations or evidence of real-world accuracy. The demand and supply curves are separate from the fitted regression. Predictions are not constrained to non-negative values, so check the quantity domain before using predictions in economic calculations.

## Main Formulas

For price P and quantity Q, the linear functions use:

- Demand: Qd(P) = a - bP
- Supply: Qs(P) = c + dP
- Total cost: TC(Q) = FC + vQ
- Total revenue: TR(P, Q) = P * Q
- Profit: Profit(Q) = TR(Q) - TC(Q)
- Marginal values: MC(Q) = dTC/dQ, MR(Q) = dTR/dQ, and MP(Q) = dProfit/dQ
- Point price elasticity: E(P) = (dQ/dP) × (P/Q)

For polynomial cost, revenue, or profit functions, MELBA calculates derivatives analytically. Profit optimization searches a finite quantity interval, evaluates real stationary points in that interval, and compares them with both endpoints. A zero first derivative alone is not treated as proof of a maximum.

The regression objective uses Huber loss on standardized targets with an L2 penalty on non-intercept coefficients. See the algorithm guide and Economic Mathematics guide for assumptions and details.

## Google Colab

Open the self-contained synthetic walkthrough in [Google Colab](https://colab.research.google.com/github/NezzyGremory/MELBA/blob/main/notebooks/MELBA_demo.ipynb). The notebook bundles a source snapshot so its regression and economics examples can run without a local repository checkout. Notebook results describe only the generated demonstration data.

## Examples and Tests

Run examples from a source checkout:

~~~bash
python examples/basic_usage.py
python examples/business_sales.py
python examples/economic_functions.py
~~~

Run the unit test suite:

~~~bash
python -m pytest
~~~

## Limitations

- Inputs must be finite numeric data; MELBA does not encode categories or impute missing values.
- The default quadratic feature expansion creates O(p²) terms for p input columns, so large feature sets can require substantial memory and computation.
- Evaluation scores describe only the supplied sample. Use a held-out dataset to assess predictive performance.
- The estimator does not provide causal estimates, time-series handling, feature selection, uncertainty intervals, or non-negative prediction constraints.
- Economic parameters and units are supplied by the caller. The cost function is linear unless a polynomial is provided.
- Profit optimization is bounded by the finite interval passed to the function; it does not claim an optimum outside those bounds.
- Elasticity describes the supplied mathematical demand function and does not establish a causal price effect.

MELBA is an inspectable learning and calculation tool, not a validated domain model for making business decisions. See the [framework guide](docs/framework.md), [algorithm guide](docs/algorithm.md), and [Economic Mathematics guide](docs/economics.md).
