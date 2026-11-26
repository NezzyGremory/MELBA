import numpy as np
import pytest

from melba.economics import DemandFunction, Elasticity, PolynomialFunction


def test_linear_demand_elasticity_matches_manual_calculation():
    demand = DemandFunction(a=100, b=2)

    result = Elasticity.price_demand(demand, price=20)

    assert result.quantity == 60
    assert result.derivative == -2
    assert result.elasticity == pytest.approx(-2 / 3)
    assert result.category == "inelastic"
    assert "less than price" in result.interpretation


def test_unitary_elasticity_and_zero_price_edge_case():
    demand = DemandFunction(a=100, b=5)

    unitary = Elasticity.price_demand(demand, price=10)
    zero_price = Elasticity.price_demand(demand, price=0)

    assert unitary.elasticity == -1
    assert unitary.category == "unitary elastic"
    assert zero_price.elasticity == 0
    assert zero_price.category == "inelastic"


def test_polynomial_demand_uses_its_analytical_derivative():
    demand = PolynomialFunction([100, -2, -0.1])

    result = Elasticity.price_demand(demand, price=10)

    assert result.quantity == 70
    assert result.derivative == -4
    assert result.elasticity == pytest.approx(-4 * 10 / 70)


@pytest.mark.parametrize("price", [-1, float("nan"), [10, 100]])
def test_elasticity_rejects_invalid_prices_and_zero_or_negative_quantity(price):
    with pytest.raises(ValueError):
        Elasticity.price_demand(DemandFunction(a=100, b=2), price=price)


def test_elasticity_requires_a_differentiable_demand_function():
    with pytest.raises(TypeError, match=r"calculate\(price\).+derivative_at\(price\)"):
        Elasticity.price_demand(object(), price=1)


def test_array_elasticity_results_include_categories_and_interpretations():
    demand = DemandFunction(a=100, b=2)

    result = Elasticity.price_demand(demand, price=[10, 40])

    np.testing.assert_allclose(result.elasticity, [-0.25, -4])
    np.testing.assert_array_equal(result.category, ["inelastic", "elastic"])
    assert result.interpretation.shape == (2,)
