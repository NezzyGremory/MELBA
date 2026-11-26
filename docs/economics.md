# Economic Mathematics in MELBA

This guide covers the Economic Mathematics Engine added in Phase 2. It is a separate toolkit from `MELBA` regression: economic functions calculate formulas supplied by the user, while the estimator learns numeric patterns from data. A prediction does not establish that changing a business input causes a change in an outcome.

The modules use NumPy and add no runtime dependency. Scalar calculations return Python floats; array calculations return NumPy arrays and use NumPy broadcasting where two inputs are involved.

## Economic Mathematics Overview

The package currently contains three areas:

- Linear functions for demand, supply, total cost, total revenue, and profit.
- Analytical derivatives for polynomial cost, revenue, and profit functions, plus bounded profit optimization.
- Point price elasticity for demand objects that provide both a calculation and an analytical derivative.

Import the public API from `melba.economics`:

```python
from melba.economics import (
    CostFunction,
    DemandFunction,
    Elasticity,
    PolynomialFunction,
    ProfitFunction,
    RevenueFunction,
    SupplyFunction,
)
```

All inputs must be finite real numbers. Prices, quantities, fixed/unit costs, and total-revenue/cost inputs must be non-negative. Profit may be negative to represent a loss. Polynomial coefficients and demand/supply parameters are finite; demand/supply parameters are also non-negative.

## Linear Economic Functions

### Demand

`DemandFunction(a, b)` implements `Qd = a - bP`, where `a` is the quantity intercept and `b` is the non-negative price coefficient.

```python
demand = DemandFunction(a=1000, b=5)
quantity = demand.calculate(price=100)  # 500
slope = demand.derivative_at(price=100)  # -5
```

The calculation returns the literal linear formula, including a negative value if price exceeds the function's choke price. Such a result is outside the economically meaningful quantity range; the library does not silently clip it to zero. Elasticity calculation requires strictly positive quantity.

### Supply

`SupplyFunction(c, d)` implements `Qs = c + dP`.

```python
supply = SupplyFunction(c=50, d=4)
quantity_supplied = supply.calculate(price=[0, 10, 20])
# array([50., 90., 130.])
```

### Cost, revenue, and profit

`CostFunction(fixed_cost, variable_cost)` treats `variable_cost` as cost per unit and implements `TC(Q) = FC + vQ`.

```python
cost = CostFunction(fixed_cost=100, variable_cost=10)
total_cost = cost.calculate(quantity=5)  # 150
```

`RevenueFunction.calculate(price, quantity)` implements `TR = P * Q`. `ProfitFunction.calculate(total_revenue, total_cost)` implements `profit = TR - TC` and can return a negative value for a loss.

```python
revenue = RevenueFunction().calculate(price=20, quantity=30)  # 600
profit = ProfitFunction().calculate(total_revenue=revenue, total_cost=450)  # 150
```

## Differential Economics

`PolynomialFunction` stores coefficients in ascending power order. For example, `[100, 10, 2]` represents `100 + 10Q + 2Q²`.

```python
total_cost = PolynomialFunction([100, 10, 2])
total_cost.calculate(5)                 # 200
total_cost.derivative().coefficients    # (10.0, 4.0), so MC(Q) = 10 + 4Q
total_cost.derivative_at(5)             # 30
```

`polynomial_derivative(function, order=1)` returns another polynomial. The marginal helpers evaluate the first derivative at the requested quantity:

```python
from melba.economics import marginal_cost, marginal_profit, marginal_revenue

marginal_cost(total_cost, 5)  # 30
total_revenue = PolynomialFunction([0, 50, -1])  # TR(Q) = 50Q - Q²
marginal_revenue(total_revenue, 10)  # 30; MR(Q) = 50 - 2Q
```

`marginal_cost` also accepts the linear `CostFunction`, whose derivative is its constant per-unit variable cost. `marginal_revenue` accepts a polynomial total-revenue function, or `RevenueFunction` with a `price` argument. In the latter case the derivative is taken with price held constant, so marginal revenue equals price. Use a polynomial total-revenue function when price varies with quantity. Marginal profit uses `PolynomialFunction` so the derivative is with respect to a clearly represented quantity variable.

### Profit optimization

`optimize_profit` requires the caller to supply a finite closed quantity interval; it does not invent a production range. It finds real roots of the analytical first derivative in that interval, evaluates the profit at each root and at both boundaries, then returns the highest candidate. The bounded interval ensures a continuous polynomial has a maximum on the search domain.

```python
# TR = 50Q - Q² and TC = 100 + 10Q + 2Q².
profit_function = PolynomialFunction([-100, 40, -3])
result = optimize_profit(profit_function, bounds=(0, 20))
result.optimal_quantities  # approximately (6.6667,)
result.maximum_profit     # approximately 33.3333
result.critical_points[0].classification  # "local_maximum"
```

An interior maximum must have a zero first derivative. A negative second derivative is sufficient to classify a non-degenerate stationary point as a local maximum; a positive one classifies a local minimum. When the second derivative is zero or numerically inconclusive, MELBA reports `"inconclusive"` rather than labeling the point a maximum. The global bounded search still compares all candidate profits. Derivative roots are computed numerically from the analytical derivative polynomial, so high-degree or ill-conditioned polynomials can have root-finding error.

If the profit polynomial is constant, `flat_objective` is `True`: every quantity in the interval has the same profit, and `optimal_quantities` contains the interval endpoints as representatives.

## Elasticity Analysis

`Elasticity.price_demand` calculates point price elasticity:

`E = (dQ/dP) * (P/Q)`

The sign is retained in `result.elasticity`; categories are based on its absolute magnitude:

- `elastic` when `|E| > 1` beyond the unit tolerance.
- `inelastic` when `|E| < 1` beyond the unit tolerance.
- `unitary elastic` when `|E|` is within `1e-9` of 1.

```python
demand = DemandFunction(a=100, b=5)
result = Elasticity.price_demand(demand, price=10)
result.quantity       # 50
result.derivative     # -5
result.elasticity     # -1
result.category       # "unitary elastic"
result.interpretation
```

The demand function must provide `calculate(price)` and `derivative_at(price)`. Both `DemandFunction` and `PolynomialFunction` provide these methods. Quantity must be strictly positive; zero or negative quantity raises `ValueError` because the formula would divide by zero or have no economic interpretation. Price zero is valid and yields zero point elasticity when quantity is positive.

The classification describes responsiveness in the supplied mathematical function. It does not establish a causal price effect, especially when the function came from a predictive ML model.

## Mathematical Foundations and Assumptions

- The linear demand and supply equations are used over non-negative prices. Demand can become negative outside its feasible range; the caller must choose a meaningful domain.
- Cost uses fixed cost plus a constant per-unit variable cost. It does not model nonlinear cost unless the user supplies a polynomial.
- Polynomial coefficients are finite real values ordered from the constant term upward. Derivatives are analytical; only polynomial root finding in profit optimization is numerical.
- Profit optimization is global only over the finite bounds passed to the function. It does not claim an optimum outside those bounds.
- Elasticity is a point measure, not an average over an interval. The demand quantity must be positive at each evaluated price.
- The functions do not infer missing prices, costs, quantities, or demand parameters.

## Usage Examples

The following example is also available as `examples/economic_functions.py` and uses only MELBA and NumPy, so it can be copied into a Google Colab cell after installing the package or opening the repository.

```python
import numpy as np
from melba.economics import (
    CostFunction,
    DemandFunction,
    Elasticity,
    PolynomialFunction,
    ProfitFunction,
    RevenueFunction,
    SupplyFunction,
    marginal_cost,
    optimize_profit,
)

demand = DemandFunction(a=1000, b=5)
supply = SupplyFunction(c=50, d=4)
prices = np.array([50.0, 100.0, 150.0])
print("demand:", demand.calculate(prices))
print("supply:", supply.calculate(prices))

cost = CostFunction(fixed_cost=100, variable_cost=10)
quantity = 30
revenue = RevenueFunction().calculate(price=20, quantity=quantity)
total_cost = cost.calculate(quantity)
print("profit:", ProfitFunction().calculate(revenue, total_cost))
print("linear marginal cost:", marginal_cost(cost, quantity))
print("elasticity:", Elasticity.price_demand(demand, price=100).elasticity)

polynomial_profit = PolynomialFunction([-100, 40, -3])
print("bounded optimum:", optimize_profit(polynomial_profit, (0, 20)))
```
