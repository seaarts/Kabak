import numpy as np
import pytest

from kabak.algos.knapsack.rounding import (
    _round_and_solve,
    _round_to_int,
    rounding_fptas,
)


@pytest.mark.parametrize(
    "nums, precision, expected, round_up",
    [
        ([1, 2, 3], 1 / 10, [0, 0, 0], False),
        ([1, 2, 3], 1 / 2, [0, 1, 1], False),
        ([1, 2, 3], 1 / 3, [0, 0, 1], False),
        ([1, 2, 3, 4], 1 / 2, [0, 1, 1, 2], False),
        ([1, 2, 3, 4], 1 / 3, [0, 0, 1, 1], False),
        ([1, 2, 3], 1 / 10, [1, 1, 1], True),
        ([1, 2, 3], 1 / 2, [1, 1, 2], True),
        ([1, 2, 3], 1 / 3, [1, 1, 1], True),
        ([1, 2, 3, 4], 1 / 2, [1, 1, 2, 2], True),
        ([1, 2, 3, 4], 1 / 3, [1, 1, 1, 2], True),
    ],
)
def test_round_to_int(nums, precision, round_up, expected):
    rounded = _round_to_int(nums, precision, round_up=round_up)

    assert np.allclose(rounded, expected)


@pytest.mark.parametrize(
    "profit, weight, budget, rounding_factor, expected",
    [
        ([], [], 1, 0.5, 0),
        ([5, 6], [1, 1], 1, 4, 5),
        ([5, 6], [1, 1], 1, 1 / 4, 6),  # no rounding
        ([5, 6, 7], [1, 1, 1], 1, 4, 5),  # rounding
        ([5, 6, 8], [1, 1, 1], 1, 4, 8),  # 8 different enough
        (list(range(100)), [1] * 100, 1, 50, 50),  # 50 is first non-zero
        (list(range(1_000)), [1] * 1_000, 1, 500, 500),  # 500 is first non-zero
        (list(range(10_000)), [1] * 10_000, 1, 5_000, 5_000),  # 5000 first non-zero
        (list(range(100000)), [1] * 100000, 1, 50000, 50000),  #  Big no problem
        (list(range(100000)), [1] * 100000, 1, 99999, 99999),
    ],
)
def test_round_and_solve(profit, weight, budget, rounding_factor, expected):
    val, sol = _round_and_solve(profit, weight, budget, rounding_factor)
    assert val == expected


@pytest.mark.parametrize(
    "profit, weight, budget, eps, optimal_val",
    [
        ([], [], 1, 0.001, 0),
        ([12], [3], 3, 0.0, 12),  # 0 rounding factor caught by _round_and_solve
        ([4, 2, 3], [3, 1, 2], 3, 0.0, 4),
        ([4, 2, 3], [3, 1, 2], 3, 0.5, 4),
        (list(range(10_000)), [1] * 10_000, 1, 0.5, 9_999),
        (list(range(10_000)), [1] * 10_000, 1, 0.1, 9_999),
        (list(range(10_000)), [1] * 10_000, 10, 0.1, sum(range(10_000 - 10, 10_000))),
        (list(range(10_000)), [1] * 10_000, 10, 0.01, sum(range(10_000 - 10, 10_000))),
    ],
)
def test_rounding_fptas(profit, weight, budget, eps, optimal_val):
    """Test FPTAS solution quality."""
    val, _ = rounding_fptas(profit, weight, budget, eps)

    assert val >= (1 - eps) * optimal_val
