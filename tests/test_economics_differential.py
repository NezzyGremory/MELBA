import numpy as np
import pytest

from melba.economics import (
    CostFunction,
    PolynomialFunction,
    RevenueFunction,
    marginal_cost,
    marginal_profit,
    marginal_revenue,
    optimize_profit,
    polynomial_derivative,
)


def test_polynomial_evaluation_and_analytical_derivatives_match_manual_values():
    total_cost = PolynomialFunction([100, 10, 2])

    assert total_cost.calculate(5) == 200
    assert polynomial_derivative(total_cost).coefficients == (10.0, 4.0)
    assert total_cost.derivative_at(5) == 30
    assert total_cost.derivative_at([0, 5]) == pytest.approx([10, 30])
    assert total_cost.derivative(order=2).coefficients == (4.0,)
    assert total_cost.derivative_at(5, order=2) == 4


def test_marginal_cost_revenue_and_profit_are_derivative_values():
    cost = PolynomialFunction([100, 10, 2])
    revenue = PolynomialFunction([0, 50, -1])
    profit = PolynomialFunction([-100, 40, -3])

    assert marginal_cost(cost, 5) == 30
    assert marginal_cost(CostFunction(fixed_cost=100, variable_cost=10), 5) == 10
    assert marginal_revenue(revenue, 10) == 30
    assert marginal_revenue(RevenueFunction(), 10, price=50) == 50
    np.testing.assert_array_equal(
        marginal_revenue(RevenueFunction(), [1, 2], price=[10, 20]),
        [10, 20],
    )
    assert marginal_profit(profit, 5) == 10


def test_profit_optimization_compares_critical_point_with_boundaries():
    # TR = 50Q - Q^2; TC = 100 + 10Q + 2Q^2; profit = -100 + 40Q - 3Q^2.
    profit = PolynomialFunction([-100, 40, -3])

    result = optimize_profit(profit, bounds=(0, 20))

    expected_quantity = 40 / 6
    assert result.optimal_quantities == pytest.approx((expected_quantity,))
    assert result.maximum_profit == pytest.approx(100 / 3)
    assert len(result.critical_points) == 1
    point = result.critical_points[0]
    assert point.quantity == pytest.approx(expected_quantity)
    assert point.second_derivative == -6
    assert point.classification == "local_maximum"
    assert result.flat_objective is False


def test_stationary_point_with_zero_second_derivative_is_not_called_a_maximum():
    profit = PolynomialFunction([0, 0, 0, 1])  # Q^3 has a stationary inflection at zero.

    result = optimize_profit(profit, bounds=(0, 2))

    assert result.critical_points[0].quantity == pytest.approx(0)
    assert result.critical_points[0].classification == "inconclusive"
    assert result.optimal_quantities == (2.0,)
    assert result.maximum_profit == 8


def test_flat_profit_reports_the_whole_interval_as_optimal_by_flag():
    result = optimize_profit(PolynomialFunction([7]), bounds=(2, 9))

    assert result.flat_objective is True
    assert result.maximum_profit == 7
    assert result.optimal_quantities == (2, 9)
    assert result.critical_points == ()


def test_profit_optimizer_keeps_distinct_critical_points_on_a_wide_interval():
    # The derivative has roots at 1 and 2 even though the search interval is large.
    profit = PolynomialFunction([0, 2, -1.5, 1 / 3])

    result = optimize_profit(profit, bounds=(0, 1e12))

    assert [point.quantity for point in result.critical_points] == pytest.approx([1, 2])


@pytest.mark.parametrize(
    "coefficients",
    [[], [1, float("nan")], [[1, 2]], ["1", "2"]],
)
def test_polynomial_coefficients_must_be_a_finite_real_vector(coefficients):
    with pytest.raises(ValueError):
        PolynomialFunction(coefficients)


def test_polynomial_rejects_nonfinite_values_invalid_derivative_order_and_wrong_function():
    function = PolynomialFunction([1, 2])

    with pytest.raises(ValueError, match="value"):
        function.calculate(float("inf"))
    with pytest.raises(ValueError, match="order"):
        function.derivative(order=-1)
    with pytest.raises(TypeError, match="derivative_at"):
        marginal_cost([100, 10], 2)
    with pytest.raises(ValueError, match="price is required"):
        marginal_revenue(RevenueFunction(), 2)


@pytest.mark.parametrize("bounds", [(-1, 10), (10, 1), (0, float("inf")), (0, 1, 2)])
def test_profit_optimization_validates_finite_nonnegative_ordered_bounds(bounds):
    with pytest.raises(ValueError):
        optimize_profit(PolynomialFunction([1, 2]), bounds=bounds)


def test_profit_optimization_requires_user_supplied_quantity_bounds():
    with pytest.raises(TypeError, match="bounds"):
        optimize_profit(PolynomialFunction([1, 2]))


def test_array_evaluation_and_derivatives_preserve_shape():
    function = PolynomialFunction([1, -3, 2])
    values = np.array([[0, 1], [2, 3]], dtype=float)

    np.testing.assert_array_equal(function.calculate(values), [[1, 0], [3, 10]])
    np.testing.assert_array_equal(function.derivative_at(values), [[-3, 1], [5, 9]])
