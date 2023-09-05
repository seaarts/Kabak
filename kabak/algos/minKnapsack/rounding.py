import numpy as np
from numpy.typing import ArrayLike

from kabak.algos.knapsack.rounding import _round_to_int
from kabak.algos.minKnapsack.dynamic_program import (
    _upper_bound,
    dynamic_program_bounded,
)


def rounding_fptas(
    cost: ArrayLike,
    weight: ArrayLike,
    demand: int,
    eps: float,
    return_sol: bool = False,
    bound_method="greedy-half",
):
    r"""
    An input rounding FPTAS for min cost knapsack.

    Notes
    -----

    This implementation uses our ``kabak.algos.minKnapsack.dynamic_program_bounded``
    together with the standard rounding approach of [Law77]_. 

    Let :math:`C_0` be a :math:`\alpha`-approximation of the optimal cost
    :math:`C^*`. We can use e.g. :math:`\alpha = 2` with
    ``kabak.algos.minKnapsack.primal_dual``.
    Define the scaling factor :math:`K` as 

    .. math::

        K = \frac{\epsilon C_0}{\alpha n}.

    Note that :math:`C_0 / \alpha \leq C^*`. The runtime is then bounded by

    .. math::

        \mathcal{O}\left(n C^* / K \right)
        = \mathcal{O}\left(\alpha n  \frac{n}{\epsilon} \frac{C^*}{C_0}\right)
        =\mathcal{O}\left(n^2 / \epsilon \right)

    Meanwhile, the error is bounded by

    .. math::

        c(S) &\leq c'(S) + K |S|  \\
        &\leq c'(S^*) + K|n| \\
        &= C^* + K |n|\\
        &= C^* + \epsilon (C_0 / \alpha) \\
        &\leq (1 + \epsilon)C^*

    where :math:`c'` indicate scalded costs, and :math:`S^*` an optimal
    solution.

    In other words, the use of the upper bounding tequique for
    ``kabak.algos.minKnapsack.bounded_dynamic_program`` makes the FPTAS
    very similar to Lawler's [Law77]_ FPTAS for Knapsack.
    """

    if (len(cost) == 0) or (len(weight) == 0):
        return -1, []

    # Get bounds
    upper_bound, lower_bound = _upper_bound(cost, weight, demand, method=bound_method)

    if upper_bound <= 0:
        return -1, []

    alpha = upper_bound / lower_bound  # approximation ratio

    # Round costs and upper bound
    K = upper_bound * eps / (len(cost) * alpha)
    cost_rounded = _round_to_int(cost, 1 / K, round_up=False)

    # Greedily select zero-cost elements
    sol = []
    residual_demand = demand

    for i, c in enumerate(cost_rounded):
        if residual_demand <= 0:
            break
        if c == 0:
            sol.append(i)
            residual_demand -= weight[i]

    # Call DP on residual instance if not done
    if residual_demand > 0:
        remaining_items = np.where(cost_rounded > 0)[0].tolist()

        _, sol_dp = dynamic_program_bounded(
            cost_rounded[cost_rounded > 0],  # Only positive cost items
            weight[cost_rounded > 0],
            residual_demand,  # Only residual demand
            upper_bound / K,  # round upper bound
            return_sol=True,
        )

        sol = sol + [remaining_items[i] for i in sol_dp]

    # Compute value and return
    val = sum([cost[item] for item in sol])

    if return_sol:
        return val, sol

    return val, []
