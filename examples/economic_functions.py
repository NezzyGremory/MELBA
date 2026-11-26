"""Examples of MELBA's economic formulas and analytical derivatives."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def main() -> None:
    demand = DemandFunction(a=1000, b=5)
    supply = SupplyFunction(c=50, d=4)
    prices = np.array([50.0, 100.0, 150.0])
    print("Demand:", demand.calculate(prices))
    print("Supply:", supply.calculate(prices))

    quantity = 30
    cost = CostFunction(fixed_cost=100, variable_cost=10)
    revenue = RevenueFunction().calculate(price=20, quantity=quantity)
    total_cost = cost.calculate(quantity)
    print("Revenue:", revenue)
    print("Total cost:", total_cost)
    print("Profit:", ProfitFunction().calculate(revenue, total_cost))
    print("Marginal cost:", marginal_cost(cost, quantity))
    print("Price elasticity at 100:", Elasticity.price_demand(demand, 100))

    # TR = 50Q - Q^2 and TC = 100 + 10Q + 2Q^2.
    profit = PolynomialFunction([-100, 40, -3])
    print("Profit optimum on [0, 20]:", optimize_profit(profit, bounds=(0, 20)))


if __name__ == "__main__":
    main()
