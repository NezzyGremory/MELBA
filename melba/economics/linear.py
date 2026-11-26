"""Linear economic functions for demand, supply, cost, revenue, and profit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import as_finite_real, as_real_values, finite_result, scalar_or_array


@dataclass(frozen=True)
class DemandFunction:
    """Linear demand ``Qd = a - bP`` for non-negative prices.

    ``a`` is the quantity intercept and ``b`` is the non-negative price
    coefficient. The raw linear formula is returned even beyond its choke
    price; negative quantities signal that the formula is outside its useful
    economic range.
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", as_finite_real(self.a, "a", nonnegative=True))
        object.__setattr__(self, "b", as_finite_real(self.b, "b", nonnegative=True))

    def calculate(self, price: object) -> float | np.ndarray:
        """Calculate quantity demanded at one price or an array of prices."""
        prices = as_real_values(price, "price", nonnegative=True)
        with np.errstate(over="ignore", invalid="ignore"):
            quantities = self.a - self.b * prices
        return scalar_or_array(finite_result(quantities, "Demand"))

    def derivative_at(self, price: object) -> float | np.ndarray:
        """Return ``dQd/dP`` at each supplied price (constant at ``-b``)."""
        prices = as_real_values(price, "price", nonnegative=True)
        return scalar_or_array(np.full_like(prices, -self.b, dtype=float))


@dataclass(frozen=True)
class SupplyFunction:
    """Linear supply ``Qs = c + dP`` for non-negative prices."""

    c: float
    d: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "c", as_finite_real(self.c, "c", nonnegative=True))
        object.__setattr__(self, "d", as_finite_real(self.d, "d", nonnegative=True))

    def calculate(self, price: object) -> float | np.ndarray:
        """Calculate quantity supplied at one price or an array of prices."""
        prices = as_real_values(price, "price", nonnegative=True)
        with np.errstate(over="ignore", invalid="ignore"):
            quantities = self.c + self.d * prices
        return scalar_or_array(finite_result(quantities, "Supply"))

    def derivative_at(self, price: object) -> float | np.ndarray:
        """Return ``dQs/dP`` at each supplied price (constant at ``d``)."""
        prices = as_real_values(price, "price", nonnegative=True)
        return scalar_or_array(np.full_like(prices, self.d, dtype=float))


@dataclass(frozen=True)
class CostFunction:
    """Linear total cost ``TC(Q) = fixed_cost + variable_cost * Q``.

    ``variable_cost`` is the variable cost per unit of output.
    """

    fixed_cost: float
    variable_cost: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixed_cost",
            as_finite_real(self.fixed_cost, "fixed_cost", nonnegative=True),
        )
        object.__setattr__(
            self,
            "variable_cost",
            as_finite_real(self.variable_cost, "variable_cost", nonnegative=True),
        )

    def calculate(self, quantity: object) -> float | np.ndarray:
        """Calculate total cost for one quantity or an array of quantities."""
        quantities = as_real_values(quantity, "quantity", nonnegative=True)
        with np.errstate(over="ignore", invalid="ignore"):
            costs = self.fixed_cost + self.variable_cost * quantities
        return scalar_or_array(finite_result(costs, "Cost"))

    def derivative_at(self, quantity: object) -> float | np.ndarray:
        """Return marginal cost ``dTC/dQ``, constant at ``variable_cost``."""
        quantities = as_real_values(quantity, "quantity", nonnegative=True)
        return scalar_or_array(np.full_like(quantities, self.variable_cost, dtype=float))


class RevenueFunction:
    """Total revenue function ``TR = price * quantity``."""

    def calculate(self, price: object, quantity: object) -> float | np.ndarray:
        """Calculate revenue; scalar and array inputs follow NumPy broadcasting."""
        prices = as_real_values(price, "price", nonnegative=True)
        quantities = as_real_values(quantity, "quantity", nonnegative=True)
        try:
            prices, quantities = np.broadcast_arrays(prices, quantities)
        except ValueError as exc:
            raise ValueError(
                "price and quantity must have broadcast-compatible shapes; "
                f"got {prices.shape} and {quantities.shape}."
            ) from exc
        with np.errstate(over="ignore", invalid="ignore"):
            revenue = prices * quantities
        return scalar_or_array(finite_result(revenue, "Revenue"))

    def derivative_at(self, quantity: object, *, price: object) -> float | np.ndarray:
        """Return ``dTR/dQ`` while holding the supplied price constant."""
        quantities = as_real_values(quantity, "quantity", nonnegative=True)
        prices = as_real_values(price, "price", nonnegative=True)
        try:
            prices, quantities = np.broadcast_arrays(prices, quantities)
        except ValueError as exc:
            raise ValueError(
                "price and quantity must have broadcast-compatible shapes; "
                f"got {prices.shape} and {quantities.shape}."
            ) from exc
        return scalar_or_array(prices)


class ProfitFunction:
    """Profit function ``profit = total_revenue - total_cost``."""

    def calculate(self, total_revenue: object, total_cost: object) -> float | np.ndarray:
        """Calculate profit or loss from non-negative revenue and total cost."""
        revenue = as_real_values(total_revenue, "total_revenue", nonnegative=True)
        costs = as_real_values(total_cost, "total_cost", nonnegative=True)
        try:
            revenue, costs = np.broadcast_arrays(revenue, costs)
        except ValueError as exc:
            raise ValueError(
                "total_revenue and total_cost must have broadcast-compatible shapes; "
                f"got {revenue.shape} and {costs.shape}."
            ) from exc
        with np.errstate(over="ignore", invalid="ignore"):
            profit = revenue - costs
        return scalar_or_array(finite_result(profit, "Profit"))
