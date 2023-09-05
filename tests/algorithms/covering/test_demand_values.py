import pytest

from kabak.algos.covering import demand_values


@pytest.mark.parametrize(
    "demand, epsilon, expected",
    [(4, 1, [1, 2, 4]), (3, 0.5, [1, 2, 3, 4]), (1, 1, [1])],
)
def test_demand_values(demand, epsilon, expected):
    """Test demand values are correct."""
    demands = demand_values(demand, epsilon)

    assert sorted(demands) == expected
