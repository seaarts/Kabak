"""
Greedy algorithms for min cost knapsack.
"""

from numpy.typing import ArrayLike


def greedy_half(cost: ArrayLike, weight: ArrayLike, budget: int) -> tuple:
    r"""
    Returns a min cost knapsack of at most twice the optimal cost.

    Notes
    -----

    Based on Csirik and Frenk's [CF91]_ simplified greedy algoirthm.

    Let there be $n$ items with cost :math"`c_j` and size :math:`a_j`. The
    items are first sorted, such that

    .. math::
      c_1 / a_1 \leq c_2 / a_2 \leq \cdots c_{n-1}/a_{n-1} \leq c_n / a_n.

    Then, a sequence of items are selected until the ``budget`` is satisfied.
    Let ``k`` be the index where this is the case. The algorithm then deletes
    the penultimate item ``k-1`` while the solution remains feasible.

    The algorithm runs in :math:`\mathcal{O}(n\log n)`-time, and returns a
    solution with cost at most 2 times the optimum.

    This is faster than a general covering greedy algorithm - it
    is used to override the greedy option for MinCostKnapsack.
    """

    val, sol = 0, []

    if len(cost) == 0 or len(weight) == 0 and budget == 0:
        return val, sol
    if len(cost) == 0 or len(weight) == 0 and budget > 0:
        return -1, sol

    residual_budget = budget

    data = [(c / w, w, i) for i, (c, w) in enumerate(zip(cost, weight))]

    for _, w, i in sorted(data):
        if residual_budget - w <= 0:
            break

        sol.append(i)
        val += cost[i]

        residual_budget -= w

    while sol and weight[sol[-1]] <= weight[i] - residual_budget:
        residual_budget += weight[sol[-1]]

        val -= cost[sol[-1]]

        sol.pop()

    sol.append(i)
    val += cost[i]

    return val, sol
