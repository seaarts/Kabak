import numpy as np
import pytest

from kabak.algos.minKnapsack import primal_dual


@pytest.mark.parametrize(
    "cost, weight, budget, return_sol, exp_val, exp_sol",
    [
        ([1, 1], [1, 1], 2, True, 2, [0, 1]),
        ([1, 1], [1, 1], 2, False, 2, []),
        ([], [], 1, True, 0, []),
        ([], [], 1, False, 0, []),
        ([1, 2], [1, 1], 1, True, 1, [0]),
        ([1, 2], [1, 1], 1, False, 1, []),
        ([1, 2], [1, 1], 9, True, -1, []),  # infeasible
        ([2, 2, 3], [1, 1, 2], 2, True, 3, [2]),
        ([2, 2, 3], [1, 1, 2], 2, False, 3, []),
        ([1, 1, 1], [1, 1, 1], 1, True, 1, [0]),  # Lexicographic tiebreak
        ([10, 10, 5], [10, 10, 1], 11, True, 20, [0, 1]),  # Fails to get OPT
        # --- Fractional inputs ----
        ([1.5, 2.5], [1.3, 1.4], 1.3, True, 1.5, [0]),
        ([2.22, 2.23], [1.3, 1.4], 1.3, True, 2.22, [0]),
        ([2.4, 1.8], [1.5, 2.5], 4.0, True, 4.2, [0, 1]),
        ([2.4, 1.8], [1.5, 2.5], 4.0, False, 4.2, []),
        ([2.4, 11.6, 1.8], [1.5, 0.4, 2.5], 4.0, True, 4.2, [0, 2]),
        ([2.1, 1.5], [1, 2], 1, True, 1.5, [1]),  # Not fooled by excess weight
        ([2.1, 1.5], [2, 2], 4, True, 3.6, [0, 1]),
        # --- Numpy array inputs ---
        (np.array([1.5, 2.5]), np.array([1.3, 1.4]), 1.3, True, 1.5, [0]),
        (np.array([2.22, 2.23]), np.array([1.3, 1.4]), 1.3, True, 2.22, [0]),
        (np.array([2.4, 11.6, 1.8]), np.array([1.5, 0.4, 2.5]), 4.0, True, 4.2, [0, 2]),
    ],
)
def test_primal_dual(cost, weight, budget, return_sol, exp_val, exp_sol):
    """Test primal dual to ensure 2-approximation."""
    val, sol = primal_dual(cost, weight, budget, return_sol=return_sol)

    assert val == exp_val and all([sorted(sol) == exp_sol])


@pytest.mark.parametrize(
    "cost, weight, budget, expected",
    [
        ([], [], 1, []),
        ([1], [1], 1, [1]),
        ([1], [1], 9, [0]),
        ([1, 1], [1, 1], 2, [1, 0]),
        ([1, 1, 1], [1, 1, 1], 3, [1, 0, 0]),
        ([1, 2], [1, 1], 2, [1, 1]),
        ([1, 3], [1, 1], 2, [1, 2]),
    ],
)
def test_primal_dual_dual(cost, weight, budget, expected):
    """Test dual variable correctness."""
    _, _, duals = primal_dual(cost, weight, budget, return_duals=True)
    assert np.allclose(np.array(expected), np.array(duals))


@pytest.mark.parametrize(
    "cost, weight, budget, expected",
    [
        ([1, 1], [1, 1], 2, 2),
        ([], [], 1, 0),
        ([1, 2], [1, 1], 1, 1),
        ([2, 2, 3], [1, 1, 2], 2, 3),
        ([1, 1, 1], [1, 1, 1], 1, 1),  # Lexicographic tiebreak
        ([10, 10, 5], [10, 10, 1], 11, 20),  # Does not get val=15
    ],
)
def test_primal_dual_val(cost, weight, budget, expected):
    """Dummy test for when solution not requested."""
    val, _ = primal_dual(cost, weight, budget)

    assert val <= expected
