"""Algorithms for the Linear Programming relaxation of {0, 1}-Knapsack."""

import numpy as np
from numpy.typing import ArrayLike


def solve_relaxation(
    profit: ArrayLike, weight: ArrayLike, budget: int, return_sol: bool = False
) -> int:
    r"""
    Return the maximum value and optionally solution to the the Knapsack LP-relaxation.

    .. todo::

      Implement Lawler's :math:`\mathcal{O}(n)`-time algorithm using median
      finding rather than sorting.

    Parameters
    ----------
    profit : Positive integer profits of items, shape ``(n_items,)``.
    weight : Positive integer weights of items, shape ``(n_imtes, )``.
    budget : Positive integer budget.
    """

    items = [(p / w, p, w, id) for id, (p, w) in enumerate(zip(profit, weight))]

    items.sort(reverse=True)

    residual_budget, value = budget, 0

    sol = np.zeros(len(items), dtype=np.float32)

    for ratio, p, w, id in items:
        if residual_budget >= w:
            value += p
            residual_budget -= w
            sol[id] = 1

        else:
            value += residual_budget * ratio
            sol[id] = residual_budget / w
            break

    if return_sol:
        return value, sol

    return value


def greedy_approx(profit: ArrayLike, weight: ArrayLike, budget: int) -> int:
    r"""
    Return a 2-factor approximation using greedy.

    .. todo::

      Implement Lawler's :math:`\mathcal{O}(n)`-time algorithm using median
      finding rather than sorting.

    Parameters
    ----------
    profit : Positive integer profits of items, shape ``(n_items,)``.
    weight : Positive integer weights of items, shape ``(n_imtes, )``.
    budget : Positive integer budget.
    """

    if not profit or not weight:
        return 0

    items = [(p / w, p, w) for p, w in zip(profit, weight)]

    items.sort(reverse=True)

    residual_budget, value = budget, 0

    for _, p, w in items:
        if residual_budget < w:
            break

        value += p
        residual_budget -= w

    return max(value, max(profit))
