import pytest

from kabak.algos.knapsack.linear import greedy_approx, solve_relaxation


@pytest.mark.parametrize(
    "profit, weight, budget, expected",
    [
        ([], [], 1, 0),
        ([1], [1], 1, 1),
        ([1], [1], 0, 0),
        ([1, 3], [1, 2], 1, 1.5),
        ([1, 3], [2, 2], 3, 3.5),
        ([1, 4, 8], [1, 3, 6], 4, 4 + 8 / 6),
    ],
)
def test_solve_relaxation_val_only(profit, weight, budget, expected):
    val = solve_relaxation(profit, weight, budget)
    assert val == expected


@pytest.mark.parametrize(
    "profit, weight, budget, opt_val",
    [
        ([], [], 1, 0),
        ([1], [1], 1, 1),
        ([1], [1], 0, 0),
        ([1, 3], [1, 2], 1, 1),
        ([1, 3], [2, 2], 3, 3),
        ([1, 4, 4], [1, 3, 4], 4, 5),
        ([2, 2, 2], [2, 2, 2], 5, 4),
        ([4, 5, 6], [2, 2, 2], 5, 11),
        ([3, 5, 6, 9], [2, 2, 2, 3], 6, 15),
    ],
)
def test_greedy_approx(profit, weight, budget, opt_val):
    val = greedy_approx(profit, weight, budget)
    assert val >= opt_val / 2
