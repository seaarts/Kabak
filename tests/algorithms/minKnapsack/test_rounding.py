import numpy as np
import pytest

from kabak.algos.minKnapsack.rounding import rounding_fptas


@pytest.mark.parametrize(
    "cost, weight, demand, eps, return_sol, opt_val, exp_sol",
    [
        ([], [], 1, 0.1, True, 0, []),
        ([], [1], 1, 0.99, True, 0, []),
        ([1], [], 1, 0.99, True, 0, []),
        ([1], [1], 10, 0.99, True, 0, []),
        ([1], [1], 10, 0.99, False, 0, []),
        ([1], [1], 1, 0.001, False, 1, []),
        ([1], [1], 1, 0.001, True, 1, [0]),
        ([1, 5], [2, 2], 2, 4.0, True, 1, [0]),
        ([1, 5], [2, 2], 2, 4.0, False, 1, []),
        ([1, 2, 5], [2, 2, 2], 3, 4.0, True, 3, [0, 1]),  # Rounded cost 0
        ([1, 2, 5], [2, 2, 2], 3, 4.0, False, 3, []),
        ([2, 1], [2, 1], 1, 50.0, True, 1, [0]),  # Hughe eps makes items equivalent
        ([2, 1], [2, 1], 1, 50.0, False, 1, []),
        ([2, 1], [2, 1], 1, 0.001, True, 1, [1]),  # Small eps -> Pick cheaper
        ([2, 1], [2, 1], 1, 0.001, False, 1, []),
        # --- Fractional inputs ---
        ([2.1, 1.5], [2, 2], 2, 0.001, True, 1.5, [1]),
        ([2.1, 1.5], [2, 2], 2, 0.001, False, 1.5, []),
        ([2.1, 1.5], [1.5, 2.5], 2.5, 0.01, True, 1.5, [1]),
        ([2.1, 1.5], [1.5, 2.5], 2.5, 0.01, False, 1.5, []),
        ([2.4, 11.6, 1.8], [1.5, 0.4, 2.5], 4.0, 0.02, True, 4.2, [0, 2]),
    ],
)
def test_rounding_fptas(cost, weight, demand, eps, return_sol, opt_val, exp_sol):
    """Test roundng fptas.

    Note that opt_val is the true optimum, while the expected
    solution is the solution one would expect from the FPTAS, not
    necessarily the optimal solution.

    Note: Mark empy instances with opt_val = 0, because scaling by eps
    confuses the checker when val is -1...
    """
    cost, weight = np.array(cost), np.array(weight)

    val, sol = rounding_fptas(cost, weight, demand, eps, return_sol=return_sol)

    print(sol)
    assert (val <= (1 + eps) * opt_val) and all([sorted(sol) == exp_sol])
