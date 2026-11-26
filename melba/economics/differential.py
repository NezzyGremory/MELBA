"""Analytical derivatives and bounded profit optimization for polynomials."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ._validation import as_real_values, finite_result, scalar_or_array
from .linear import CostFunction, RevenueFunction


@dataclass(frozen=True, init=False)
class PolynomialFunction:
    """A polynomial whose coefficients are ordered from constant upward.

    For example, ``PolynomialFunction([100, 10, 2])`` represents
    ``100 + 10Q + 2Q**2``. Inputs and outputs must be finite real numbers.
    """

    coefficients: tuple[float, ...]

    def __init__(self, coefficients: object) -> None:
        values = as_real_values(coefficients, "coefficients")
        if values.ndim != 1:
            raise ValueError("coefficients must be a one-dimensional sequence.")
        coefficients_tuple = tuple(float(value) for value in values)
        while len(coefficients_tuple) > 1 and coefficients_tuple[-1] == 0.0:
            coefficients_tuple = coefficients_tuple[:-1]
        object.__setattr__(self, "coefficients", coefficients_tuple)

    @property
    def degree(self) -> int:
        """Return the polynomial's degree (zero for a constant polynomial)."""
        return len(self.coefficients) - 1

    def calculate(self, value: object) -> float | np.ndarray:
        """Evaluate the polynomial using NumPy's ascending-coefficient convention."""
        values = as_real_values(value, "value")
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.polynomial.polynomial.polyval(values, self.coefficients)
        result_array = np.asarray(result, dtype=float)
        return scalar_or_array(finite_result(result_array, "Polynomial"))

    def derivative(self, order: int = 1) -> "PolynomialFunction":
        """Return the analytical derivative polynomial of the requested order."""
        if isinstance(order, bool) or not isinstance(order, Integral) or order < 0:
            raise ValueError("order must be a non-negative integer.")
        values = np.asarray(self.coefficients, dtype=float)
        for _ in range(int(order)):
            if values.size == 1:
                values = np.zeros(1, dtype=float)
            else:
                values = values[1:] * np.arange(1, values.size, dtype=float)
        return PolynomialFunction(values)

    def derivative_at(self, value: object, order: int = 1) -> float | np.ndarray:
        """Evaluate the analytical derivative at one or more values."""
        return self.derivative(order).calculate(value)


@dataclass(frozen=True)
class CriticalPoint:
    """A stationary quantity and its second-derivative classification."""

    quantity: float
    profit: float
    second_derivative: float
    classification: str


@dataclass(frozen=True)
class ProfitOptimizationResult:
    """Global maximum of a polynomial profit function on finite bounds."""

    bounds: tuple[float, float]
    maximum_profit: float
    optimal_quantities: tuple[float, ...]
    critical_points: tuple[CriticalPoint, ...]
    flat_objective: bool = False


def _require_polynomial(function: object, name: str) -> PolynomialFunction:
    if not isinstance(function, PolynomialFunction):
        raise TypeError(f"{name} must be a PolynomialFunction.")
    return function


def polynomial_derivative(
    function: PolynomialFunction,
    order: int = 1,
) -> PolynomialFunction:
    """Return the analytical derivative of a ``PolynomialFunction``."""
    return _require_polynomial(function, "function").derivative(order)


def marginal_cost(
    cost_function: PolynomialFunction | CostFunction,
    quantity: object,
) -> float | np.ndarray:
    """Evaluate marginal cost for a polynomial or linear total-cost function."""
    derivative_at = getattr(cost_function, "derivative_at", None)
    if not callable(derivative_at):
        raise TypeError("cost_function must provide derivative_at(quantity).")
    return derivative_at(quantity)


def marginal_revenue(
    revenue_function: PolynomialFunction | RevenueFunction,
    quantity: object,
    *,
    price: object | None = None,
) -> float | np.ndarray:
    """Evaluate marginal revenue for a polynomial or fixed-price revenue function.

    ``price`` is required for ``RevenueFunction``, where the derivative holds
    price constant. Polynomial total-revenue functions already express price
    as part of the quantity relationship and need no separate price argument.
    """
    if isinstance(revenue_function, RevenueFunction):
        if price is None:
            raise ValueError("price is required for RevenueFunction marginal revenue.")
        return revenue_function.derivative_at(quantity, price=price)
    if price is not None:
        raise ValueError("price must be omitted for a polynomial revenue function.")
    revenue = _require_polynomial(revenue_function, "revenue_function")
    return revenue.derivative_at(quantity)


def marginal_profit(
    profit_function: PolynomialFunction,
    quantity: object,
) -> float | np.ndarray:
    """Evaluate marginal profit ``dProfit/dQ`` for a polynomial profit function."""
    profit = _require_polynomial(profit_function, "profit_function")
    return profit.derivative_at(quantity)


def _real_roots_in_bounds(
    derivative: PolynomialFunction,
    lower: float,
    upper: float,
) -> list[float]:
    coefficients = np.asarray(derivative.coefficients, dtype=float)
    if coefficients.size == 1:
        return []
    try:
        roots = np.polynomial.polynomial.polyroots(coefficients)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ArithmeticError(
            "Could not resolve derivative roots; rescale the polynomial or use narrower bounds."
        ) from exc
    bound_scale = max(1.0, abs(lower), abs(upper))
    interval_tolerance = 64 * np.finfo(float).eps * bound_scale
    real_roots: list[float] = []
    for root in roots:
        real = float(np.real(root))
        imaginary = float(np.imag(root))
        if abs(imaginary) > 1e-9 * max(1.0, abs(real)):
            continue
        if lower - interval_tolerance <= real <= upper + interval_tolerance:
            real_roots.append(min(upper, max(lower, real)))
    real_roots.sort()
    distinct: list[float] = []
    for root in real_roots:
        root_tolerance = 1e-8 * max(1.0, abs(root), abs(distinct[-1]) if distinct else 0.0)
        if not distinct or abs(root - distinct[-1]) > root_tolerance:
            distinct.append(root)
    return distinct


def _classify_stationary_point(function: PolynomialFunction, quantity: float) -> tuple[float, str]:
    second = function.derivative_at(quantity, order=2)
    second_value = float(second)
    second_coefficients = np.abs(np.asarray(function.derivative(2).coefficients))
    with np.errstate(over="ignore", invalid="ignore"):
        scale = float(np.polynomial.polynomial.polyval(abs(quantity), second_coefficients))
    if not np.isfinite(scale):
        return second_value, "inconclusive"
    tolerance = 1e-10 * max(1.0, scale)
    if second_value < -tolerance:
        return second_value, "local_maximum"
    if second_value > tolerance:
        return second_value, "local_minimum"
    return second_value, "inconclusive"


def optimize_profit(
    profit_function: PolynomialFunction,
    bounds: tuple[float, float],
) -> ProfitOptimizationResult:
    """Find a global profit maximum on a finite, closed quantity interval.

    All real roots of the analytical first derivative in the interval are
    compared with both boundaries. A stationary point is labeled a local
    maximum or minimum only when the second-derivative test is decisive; a
    zero or numerically inconclusive curvature is reported as inconclusive.
    """
    function = _require_polynomial(profit_function, "profit_function")
    validated_bounds = as_real_values(bounds, "bounds", nonnegative=True)
    if validated_bounds.ndim != 1 or validated_bounds.size != 2:
        raise ValueError("bounds must be a pair of finite non-negative quantities.")
    lower, upper = (float(value) for value in validated_bounds)
    if lower > upper:
        raise ValueError("bounds must be ordered as (lower_quantity, upper_quantity).")

    if function.degree == 0:
        optimal = (lower,) if lower == upper else (lower, upper)
        return ProfitOptimizationResult(
            bounds=(lower, upper),
            maximum_profit=function.coefficients[0],
            optimal_quantities=optimal,
            critical_points=(),
            flat_objective=True,
        )

    derivative = function.derivative()
    roots = _real_roots_in_bounds(derivative, lower, upper)
    critical_points: list[CriticalPoint] = []
    for quantity in roots:
        curvature, classification = _classify_stationary_point(function, quantity)
        profit = float(function.calculate(quantity))
        critical_points.append(
            CriticalPoint(
                quantity=quantity,
                profit=profit,
                second_derivative=curvature,
                classification=classification,
            )
        )

    candidates = [lower]
    for quantity in [*roots, upper]:
        if not any(
            abs(quantity - existing)
            <= 64 * np.finfo(float).eps * max(1.0, abs(quantity), abs(existing))
            for existing in candidates
        ):
            candidates.append(quantity)
    values = [float(function.calculate(quantity)) for quantity in candidates]
    maximum = max(values)
    comparison_tolerance = 1e-10 * max(1.0, abs(maximum))
    optimal = tuple(
        quantity
        for quantity, value in zip(candidates, values)
        if abs(value - maximum) <= comparison_tolerance
    )
    return ProfitOptimizationResult(
        bounds=(lower, upper),
        maximum_profit=maximum,
        optimal_quantities=optimal,
        critical_points=tuple(critical_points),
        flat_objective=False,
    )
