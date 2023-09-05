import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from kabak.algos.knapsack.dynamic_program import optimal_solution
from kabak.algos.knapsack.linear import greedy_approx


def _round_to_int(
    nums: ArrayLike,
    precision: float,
    round_up: bool = False,
    dtype: DTypeLike = np.int32,
) -> ArrayLike:
    """
    Round fractional inputs to integers with given precision.

    Returns ``floor(nums * precision)``, with the option of dividing each output
    by the greatest common multiple (``gcm``).

    Parameters
    ----------
    nums : Array of values to be rounded.
    precision : Determines the coarseness of the rounding.
        Larger values imply a finer precision.
    dtype : Specifies the output array datatype.

    """

    nums = np.array(nums)  # avoid list-multiplication

    if round_up:
        rounded = np.ceil(nums * precision)  # / precision
    else:
        rounded = np.floor(nums * precision)  # / precision

    return rounded.astype(dtype)


def _round_and_solve(
    profit: ArrayLike, weight: ArrayLike, budget: int, rounding_factor: float
):
    """
    Solve Knapsack on rounded inputs. Return unrounded value of selection.

    Prameters
    ---------
    profit : Array of positive integral profits.
    weight : Array of positive integral weights.
    budget : Maximum permissible weight in knapsack.
    rounding_factor : Positive value with which to divide inputs by.
    """

    if rounding_factor <= 1:
        return optimal_solution(profit, weight, budget)

    rounded_profit = _round_to_int(profit, 1 / rounding_factor, round_up=False)

    _, sol = optimal_solution(rounded_profit, weight, budget)

    return int(sum(profit[i] for i in sol)), sol


def rounding_fptas(profit: ArrayLike, weight: ArrayLike, budget: int, eps: float):
    """
    Knapsack FTPAS via greedy approximation of upper bound.

    Based on Lawler's 1977 [Law77]_ rounding algorithm.

    The algorithm requires a fast approximation to get a reasonably tight
    upper bound. This implementation fills this role with the
    ``greedy_approx``-algorithm. This is a 2-approximation, so that
    twice the approximate value, ``2 * greedy_approx``,  is an upper bound
    on the optimal value. This ensures proper rounding while at most doubing
    the runtime.

    """

    if not profit or not weight:
        return 0, []

    # Can be substituted for different APX algo / ratio
    greedy_val = greedy_approx(profit, weight, budget)
    approx_ratio = 2

    rounding_factor = greedy_val * approx_ratio * eps / len(profit)

    return _round_and_solve(profit, weight, budget, rounding_factor)
