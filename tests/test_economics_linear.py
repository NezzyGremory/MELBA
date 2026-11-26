import numpy as np
import pytest

from melba.economics import (
    CostFunction,
    DemandFunction,
    ProfitFunction,
    RevenueFunction,
    SupplyFunction,
)


def test_linear_demand_matches_formula_for_scalar_and_array_inputs():
    demand = DemandFunction(a=1000, b=5)

    assert demand.calculate(100) == 500
    np.testing.assert_array_equal(demand.calculate([0, 100, 200]), [1000, 500, 0])
    np.testing.assert_array_equal(demand.derivative_at([0, 100]), [-5, -5])
    assert demand.calculate(250) == -250  # raw linear formula beyond its choke price


def test_linear_supply_matches_formula():
    supply = SupplyFunction(c=50, d=4)

    assert supply.calculate(10) == 90
    np.testing.assert_array_equal(supply.calculate([0, 10, 20]), [50, 90, 130])
    assert supply.derivative_at(10) == 4


def test_linear_cost_uses_per_unit_variable_cost():
    cost = CostFunction(fixed_cost=100, variable_cost=10)

    assert cost.calculate(5) == 150
    np.testing.assert_array_equal(cost.calculate([0, 1, 5]), [100, 110, 150])


def test_revenue_and_profit_match_manual_calculations_and_broadcast():
    revenue = RevenueFunction()
    profit = ProfitFunction()

    assert revenue.calculate(20, 30) == 600
    np.testing.assert_array_equal(
        revenue.calculate(np.array([[10], [20]]), np.array([1, 2, 3])),
        [[10, 20, 30], [20, 40, 60]],
    )
    assert profit.calculate(600, 450) == 150
    assert profit.calculate(400, 450) == -50
    np.testing.assert_array_equal(profit.calculate([600, 400], [450, 450]), [150, -50])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DemandFunction(a=-1, b=2),
        lambda: DemandFunction(a=10, b=float("inf")),
        lambda: SupplyFunction(c=1, d=-2),
        lambda: CostFunction(fixed_cost=-1, variable_cost=2),
        lambda: CostFunction(fixed_cost=1, variable_cost=True),
    ],
)
def test_linear_function_parameters_are_validated(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("price", [-1, float("nan"), float("inf"), "high", []])
def test_demand_rejects_invalid_price(price):
    with pytest.raises(ValueError):
        DemandFunction(10, 1).calculate(price)


def test_supply_and_cost_reject_negative_inputs():
    with pytest.raises(ValueError, match="price"):
        SupplyFunction(1, 1).calculate(-1)
    with pytest.raises(ValueError, match="quantity"):
        CostFunction(10, 2).calculate(-1)


def test_revenue_and_profit_validate_broadcasting_and_nonnegative_inputs():
    revenue = RevenueFunction()
    profit = ProfitFunction()

    with pytest.raises(ValueError, match="non-negative"):
        revenue.calculate(-1, 3)
    with pytest.raises(ValueError, match="broadcast-compatible"):
        revenue.calculate([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="total_cost"):
        profit.calculate(10, -1)


def test_arithmetic_overflow_is_reported():
    with pytest.raises(OverflowError, match="Revenue calculation"):
        RevenueFunction().calculate(1e308, 2)
