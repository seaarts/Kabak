import numpy as np
import pytest

from kabak.algos.linearProgram.ortools import linear_program_ortools


@pytest.mark.parametrize(
    # The integers 0, 13 are solvers - see ortools documentation.
    "c, A, b, B, f, d, integral, minimize, solver_type, exp_sol, exp_val",
    [
        ([1], [[1]], [1], None, None, [1], False, True, 0, [1.0], 1.0),
        ([1], [[1]], [2], None, None, [2], False, True, 0, [2.0], 2.0),
        ([5], [[1]], [2], None, None, [2], False, True, 0, [2.0], 10.0),
        ([5], [[1]], [2], None, None, [1], False, True, 0, [], -1),
        ([2], [[2]], [3], None, None, [2], False, True, 0, [1.5], 3.0),
        ([2], [[2]], [3], None, None, [2], True, True, 14, [2.0], 4.0),
        ([2], [[2]], [3], None, None, [2], True, True, 0, [1.5], 3.0),
        ([2], None, None, [[1]], [2], [2], False, False, 0, [2.0], 4.0),
        ([2], None, None, [[2]], [3], [2], False, False, 0, [1.5], 3.0),
        ([2], None, None, [[2]], [3], [2], True, False, 14, [1.0], 2.0),
        ([2], [[2]], [3], None, None, [1], True, True, 14, [], -1),
        ([5], [[2]], [3], None, None, [2], True, True, 14, [2.0], 10.0),
        ([1], [[1]], [2], [[1]], [2], [5], False, True, 0, [2.0], 2.0),
        ([5, 1], [[2, 1]], [3], None, None, [2, 2], True, True, 14, [1.0, 1.0], 6.0),
        ([5, 1], [[2, 1]], [3], None, None, [3, 3], True, True, 14, [0.0, 3.0], 3.0),
    ],
)
@pytest.mark.skip(reason="Needs review.")
def test_ortools(c, A, b, B, f, d, integral, minimize, solver_type, exp_sol, exp_val):
    """Convert to numpy first."""

    A = np.array(A)
    B = np.array(B)

    val, sol, _ = linear_program_ortools(
        c, A, b, B, f, d, integral=integral, minimize=minimize, solver_type=solver_type
    )
    assert val == exp_val and all([sol == exp_sol])
