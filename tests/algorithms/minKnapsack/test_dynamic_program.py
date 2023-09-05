import numpy as np
import pytest

from kabak.algos.minKnapsack.dynamic_program import (
    _merge_pairs,
    _upper_bound,
    dynamic_program,
    dynamic_program_bounded,
)


@pytest.mark.parametrize(
    "cost, weight, budget, return_sol, bound_method, exp_val, exp_sol",
    [
        ([], [], 1, True, "primal-dual", -1, []),
        ([1], [1], 1, True, "primal-dual", 1, [0]),
        ([1], [2], 2, True, "primal-dual", 1, [0]),
        ([], [], 2, False, "primal-dual", -1, []),
        ([1], [2], 2, False, "primal-dual", 1, []),
        (np.array([1]), np.array([1]), 1, True, "primal-dual", 1, [0]),
        (np.array([1, 2]), np.array([1, 1]), 2, True, "primal-dual", 3, [0, 1]),
        (np.array([6, 2, 2]), np.array([5, 2, 4]), 6, True, "primal-dual", 4, [1, 2]),
        # ------ Greedy APX ------
        ([], [], 1, True, "greedy-half", -1, []),
        ([1], [1], 1, True, "greedy-half", 1, [0]),
        ([1], [2], 2, True, "greedy-half", 1, [0]),
        ([], [], 2, False, "greedy-half", -1, []),
        ([1], [2], 2, False, "greedy-half", 1, []),
        (np.array([1]), np.array([1]), 1, True, "greedy-half", 1, [0]),
        (np.array([1, 2]), np.array([1, 1]), 2, True, "greedy-half", 3, [0, 1]),
        (np.array([6, 2, 2]), np.array([5, 2, 4]), 6, True, "greedy-half", 4, [1, 2]),
    ],
)
def test_dynamic_program(
    cost, weight, budget, return_sol, bound_method, exp_val, exp_sol
):
    """Test the DP wrapper."""
    val, sol = dynamic_program(
        cost, weight, budget, return_sol=return_sol, bound_method=bound_method
    )

    assert val == exp_val and all([sorted(sol) == exp_sol])


@pytest.mark.parametrize(
    "pairs1, pairs2, expected",
    [
        ([(0, 0)], [], [(0, 0)]),
        ([], [(0, 0)], [(0, 0)]),
        ([], [], []),
        ([(0, 0)], [(1, 1)], [(0, 0), (1, 1)]),
        ([(0, 0, "a")], [(0, 0, "b")], [(0, 0, "a")]),
        ([(0, 0), (1, 1)], [(0, 0), (1, 1)], [(0, 0), (1, 1)]),
        ([(2, 5)], [(1, 1)], [(1, 1), (2, 5)]),
        ([(2, 5)], [(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 5)]),
        ([(0, 0), (2, 2)], [(1, 1), (3, 3)], [(0, 0), (1, 1), (2, 2), (3, 3)]),
        ([(0, 0), (3, 3)], [(1, 1), (3, 4)], [(0, 0), (1, 1), (3, 4)]),
        ([(1, 4, "a"), (2, 5, "b")], [(2, 2, "c")], [(1, 4, "a"), (2, 5, "b")]),
        ([(5, 5)], [(4, 6)], [(4, 6)]),
    ],
)
def test_merge_pairs(pairs1, pairs2, expected):
    """Compare output optimum value with expected value."""
    merged = _merge_pairs(pairs1, pairs2)
    print(merged)
    assert all([merged == expected])


@pytest.mark.parametrize(
    "cost, weight, demand, upper_bound, expected",
    [
        ([], [], 1, 1, []),
        ([1], [1], 1, 1, [0]),
        ([1, 1], [1, 1], 4, 4, []),  # infeaasible
        ([1, 1], [1, 1], 2, 2, [0, 1]),
        ([1, 2], [1, 2], 2, 3, [1]),
        ([2, 3], [2, 3], 2, 5, [0]),
        ([4, 2, 3], [4, 2, 3], 5, 10, [1, 2]),
        ([5, 3, 6], [2, 1, 3], 3, 15, [2]),
        ([2, 5, 6, 4], [4, 5, 6, 3], 12, 20, [0, 1, 3]),
        ([4, 5, 5, 2], [1, 5, 6, 3], 12, 20, [1, 2, 3]),
        # ([0, 1] * 100, [1, 1] * 100, 100, list(range(1, 200, 2))),  # odd ids only
    ],
)
def test_dynamic_program_bounded_sol(cost, weight, demand, upper_bound, expected):
    """Compare output list of indices with expected list of indices."""
    _, sol = dynamic_program_bounded(
        cost, weight, demand, upper_bound=upper_bound, return_sol=True
    )
    assert all([sorted(sol) == expected])  # sort b/c item order is shuffled by alg


@pytest.mark.parametrize(
    "cost, weight, demand, upper_bound, expected_val",
    [
        ([], [], 1, 1, -1),
        ([1], [1], 1, 1, 1),
        ([1], [1], 4, 4, -1),  # infeasible
        ([1, 1], [1, 1], 2, 2, 2),
        ([1, 2], [1, 2], 2, 5, 2),
        ([2, 3], [2, 3], 2, 5, 2),
        ([4, 2, 3], [4, 2, 3], 5, 10, 5),
        ([5, 3, 6], [2, 1, 3], 3, 20, 6),
        ([2, 5, 6, 4], [4, 5, 6, 3], 12, 20, 11),
        ([4, 5, 5, 2], [1, 5, 6, 3], 12, 20, 12),
    ],
)
def test_dynamic_program_bounded_val(cost, weight, demand, upper_bound, expected_val):
    """Compare output list of indices with expected list of indices."""
    val, _ = dynamic_program_bounded(
        cost, weight, demand, upper_bound, return_sol=False
    )
    assert val == expected_val  # sort b/c item order is shuffled by alg


@pytest.mark.parametrize(
    "cost, weight, demand, expected",
    [
        ([1], [1], 1, 1),
        ([1, 2, 3], [3, 2, 1], 3, 1),
        ([3, 2, 2], [2, 4, 3], 5, 4),
        ([1] * 100, [1] * 100, 10, 10),
    ],
)
def test_upper_bound(cost, weight, demand, expected):
    """Make sure a valid upper and lower bound are returned."""
    upp, low = _upper_bound(cost, weight, demand)

    assert upp >= expected and low <= expected


@pytest.mark.parametrize(
    "cost, weight, budget, method",
    [([1], [1], 1, "gronky"), ([1], [1], 1, "dula-pipa"), ([1], [1], 1, "dua-lipa")],
)
def test_upper_bound_bad_val(cost, weight, budget, method):
    with pytest.raises(ValueError):
        _, _ = _upper_bound(cost, weight, budget, method)


# @pytest.mark.parametrize(
#    "cost, weight, budget, method",
#    [
#        ([1], [1], 1, "greedy-half"),
#        ([1], [1], 1, "greedy-half"),
#        ([1], [1], 1, "greedy-half"),
#    ],
# )
# def test_upper_bound_greedy(cost, weight, budget, method):
#    """Remove if Greedy-half is implemented!"""
#    with pytest.raises(NotImplementedError):
#        _, _ = _upper_bound(cost, weight, budget, method)
