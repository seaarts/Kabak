import numpy as np
import pytest

from kabak.algos.knapsack.dynamic_program import (
    _merge_pairs,
    optimal_solution,
    optimal_value,
)


@pytest.mark.parametrize(
    "pairs1, pairs2, expected",
    [
        ([(0, 0)], [], [(0, 0)]),
        ([], [(0, 0)], [(0, 0)]),
        ([], [], []),
        ([(0, 0)], [(1, 1)], [(0, 0), (1, 1)]),
        ([(0, 0)], [(0, 0)], [(0, 0)]),
        ([(0, 0), (1, 1)], [(0, 0), (1, 1)], [(0, 0), (1, 1)]),
        ([(2, 5)], [(1, 1)], [(1, 1), (2, 5)]),
        ([(2, 5)], [(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 5)]),
        ([(0, 0), (2, 2)], [(1, 1), (3, 3)], [(0, 0), (1, 1), (2, 2), (3, 3)]),
        ([(0, 0), (2, 3)], [(1, 1), (2, 3)], [(0, 0), (1, 1), (2, 3)]),
    ],
)
def test_merge_pairs(pairs1, pairs2, expected):
    """Compare output optimum value with expected value."""
    merged = _merge_pairs(pairs1, pairs2)
    print(merged)
    assert all([merged == expected])


@pytest.mark.parametrize(
    "profit, weight, budget, expected",
    [
        ([1], [1], 1, 1),
        ([1, 1], [1, 1], 1, 1),
        ([1, 2], [1, 2], 2, 2),
        ([2, 3], [2, 3], 2, 2),
        ([1] * 10000, [1] * 10000, 10, 10),
    ],
)
def test_optimal_value(profit, weight, budget, expected):
    """Dummy test for when solution is not requested."""
    val = optimal_value(profit, weight, budget)
    assert val == expected


@pytest.mark.parametrize(
    "profit, weight, budget, expected",
    [
        ([1], [1], 1, [0]),
        ([1, 1], [1, 1], 2, [0, 1]),
        ([1, 2], [1, 2], 2, [1]),
        ([2, 3], [2, 3], 2, [0]),
        ([1] * 10000, [1] * 10000, 10, list(range(10))),
        ([4, 2, 3], [4, 2, 3], 5, [1, 2]),
        ([5, 3, 6], [2, 1, 3], 3, [0, 1]),
        ([2, 5, 5, 4], [4, 5, 6, 3], 12, [0, 1, 3]),
        ([4, 5, 5, 2], [1, 5, 6, 3], 12, [0, 1, 2]),
        ([0, 1] * 100, [1, 1] * 100, 100, list(range(1, 200, 2))),  # odd ids only
    ],
)
def test_optimal_solution(profit, weight, budget, expected):
    """Compare output list of indices with expected list of indices."""
    _, sol = optimal_solution(profit, weight, budget)
    assert all([sorted(sol) == expected])  # sort b/c item order is shuffled by alg
