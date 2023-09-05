import pytest

from kabak.algos.minKnapsack import greedy_half


@pytest.mark.parametrize(
    "cost, weight, budget, expected",
    [
        ([1, 1, 1, 1], [1, 1, 1, 1], 2, [0, 1]),
        ([1, 1, 1, 3], [1, 1, 1, 3], 3, [0, 1, 2]),
        ([], [], 0, []),
        ([], [], 1, []),
        ([10], [10], 10, [0]),
        ([2, 3, 3, 4], [2, 3, 3, 4], 6, [0, 1, 2]),
        ([1, 1, 2, 5], [10, 5, 10, 5], 20, [0, 2]),
        ([1, 1, 1, 3, 1], [10, 5, 5, 15, 1], 25, [0, 3]),
    ],
)
def test_greedy_half(cost, weight, budget, expected):
    """Test greedy algorithm solutions."""

    _, sol = greedy_half(cost, weight, budget)

    assert all([sorted(sol) == expected])
