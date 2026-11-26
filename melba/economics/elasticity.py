"""Price-elasticity calculations for differentiable demand functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import as_real_values, finite_result, scalar_or_array


@dataclass(frozen=True)
class ElasticityResult:
    """Point price elasticity and its magnitude-based category."""

    price: float | np.ndarray
    quantity: float | np.ndarray
    derivative: float | np.ndarray
    elasticity: float | np.ndarray
    category: str | np.ndarray

    @property
    def interpretation(self) -> str | np.ndarray:
        """Explain how quantity responds proportionally to price."""
        explanations = {
            "elastic": "Quantity demanded changes proportionally more than price.",
            "inelastic": "Quantity demanded changes proportionally less than price.",
            "unitary elastic": "Quantity demanded changes in the same proportion as price.",
        }
        if isinstance(self.category, str):
            return explanations[self.category]
        return np.select(
            [self.category == "elastic", self.category == "unitary elastic"],
            [explanations["elastic"], explanations["unitary elastic"]],
            default=explanations["inelastic"],
        )


class Elasticity:
    """Price-elasticity calculations for linear or polynomial demand functions."""

    @staticmethod
    def price_demand(demand_function: object, price: object) -> ElasticityResult:
        """Calculate ``(dQ/dP) * (P/Q)`` and interpret its absolute magnitude.

        ``demand_function`` must provide ``calculate(price)`` and
        ``derivative_at(price)`` methods. Quantity must be strictly positive at
        every requested price. Categories use a relative and absolute
        tolerance of ``1e-9`` around unit elasticity.
        """
        calculate = getattr(demand_function, "calculate", None)
        derivative_at = getattr(demand_function, "derivative_at", None)
        if not callable(calculate) or not callable(derivative_at):
            raise TypeError(
                "demand_function must provide calculate(price) and derivative_at(price)."
            )

        prices = as_real_values(price, "price", nonnegative=True)
        quantities = as_real_values(calculate(prices), "demand quantity")
        derivatives = as_real_values(derivative_at(prices), "demand derivative")
        try:
            prices, quantities, derivatives = np.broadcast_arrays(
                prices, quantities, derivatives
            )
        except ValueError as exc:
            raise ValueError(
                "Demand quantity and derivative must be broadcast-compatible with price."
            ) from exc
        if np.any(quantities <= 0):
            raise ValueError(
                "Demand quantity must be greater than zero to calculate price elasticity."
            )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            elasticity = derivatives * (prices / quantities)
        elasticity = finite_result(elasticity, "Price elasticity")
        magnitude = np.abs(elasticity)
        unitary = np.isclose(magnitude, 1.0, rtol=1e-9, atol=1e-9)
        categories = np.where(
            unitary,
            "unitary elastic",
            np.where(magnitude > 1, "elastic", "inelastic"),
        )

        if elasticity.ndim == 0:
            category: str | np.ndarray = str(categories.item())
        else:
            category = categories
        return ElasticityResult(
            price=scalar_or_array(prices),
            quantity=scalar_or_array(quantities),
            derivative=scalar_or_array(derivatives),
            elasticity=scalar_or_array(elasticity),
            category=category,
        )
