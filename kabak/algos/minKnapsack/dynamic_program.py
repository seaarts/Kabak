r"""
A dynamic programming implementation for minimum cost knapsack.
"""

import numpy as np
from numpy.typing import ArrayLike

from kabak.algos.minKnapsack.greedy import greedy_half
from kabak.algos.minKnapsack.primal_dual import primal_dual
from kabak.structures.graph import TreeNode


def dynamic_program(
    cost: ArrayLike,
    weight: ArrayLike,
    demand: float,
    return_sol: bool = False,
    bound_method: str = "primal-dual",
) -> tuple:
    """
    Returns the optimum to a min cost knapsack problem.
    """
    if len(cost) == 0 or len(weight) == 0:
        return -1, []

    bound, _ = _upper_bound(cost, weight, demand, method=bound_method)

    return dynamic_program_bounded(
        cost, weight, demand, upper_bound=bound, return_sol=return_sol
    )


def dynamic_program_bounded(
    cost: ArrayLike,
    weight: ArrayLike,
    demand: int,
    upper_bound: int,
    return_sol: bool = False,
) -> tuple:
    r"""
    Return the minimum cost knapsack value.

    Parameters
    ----------

    cost : ArrayLike
        Array of positive integer costs for each item. Shape ``(n_items,)``.
    weight : ArrayLike
        Array of positive integer weights for each item. Shape ``(n_items)``.
    demand : int
        Positive integer demand indicating minimum permissible knapsack weight.
    upper_bound: int
        An upper bound on the cost of an optimal solution. Ideally this is based
        on a constant factor approximation, e.g.
        ``algos.MinKnapsack.primal_dual()``.
    upper_bound: int
        An upper bound on the minimum cost.
    return_sol: bool
        Whether to return the solution as well as the solutio value.

    Notes
    -----

    This algorithm is adapted from [Law77]_ with some new ideas and additions.

    The main idea is to maintain and expand on a list of cost-weight ``pairs``
    ``(C, W)``. Initially thre is only the pair ``(0, 0)``.
    Then, the list of items is iterated over once.
    When item ``i`` is considered, each pair ``(C, W)`` generates a new pair
    ``(C + cost[i], W + weight[i])``. Feasible pairs have ``W >= D``. The
    optimal value is the smallest cost whose corresponding weight is feasible.

    There are ways to ensure that not too many pairs are generated. In the
    maximum Knapsack setting, Lawler uses an elegant dominance argument to
    prune the number of profit-weight pairs. A pair is said to be *dominated*
    if there exists another pair with higher profit and no higher weight.
    Pruning (weakly) dominated pairs ensures that there is at most one pair
    for every weight and profit value. This ensures that are at most
    :math:`\min\{P^*, b\}` pairs, where :math:`b` is the budget. By maintaining
    a bounded number of pairs, the workload of the algorithm is kept bounded.
    In the minimum cost knapsack problem, however, bounding the number of pairs
    less straightforward.

    There are two natural approaches to construct cost-weight pairs for
    minimum cost knapsak. The first is to start from the maximal feasible
    solution that includes all items. This solution candidate is feasible
    (if not, the instance has no solution), however may have an excessively
    high cost. One can then gradually generate cost-weight pairs by removing
    items one by one. A problem is that there can be up to
    :math:`\min\{C_{total} - C^*, W_{total} - W^*\}` pairs, where :math:`X^*`
    denotes the value of an optimal solution. This makes bounding the runtime
    challenging.

    Another natural approach is to start from an empty, infeasible,
    solution with pair ``(0,0)`` and gradually add items. This approach,
    however, also has challenges with respect to bounding the number of
    pairs. Before all items have been cosidered, there is no trivial way to
    discard pairs, except for dominance. In the maxium knapsack model the
    budget constraint could be employed whenever adding an item would make
    the solution too heave. However, in for min-cost knapsack it is not safe
    to discard a pair once the weight demand is satisfied, although it might
    be safe to stop adding items to the pair. An difficult instance is one
    in which all items have :math:`D` weight, but decreasing cost. A naive
    implementation would need to form all :math:`1 + 2 + \cdots + n` pairs.
    This makes the runtime independent of the size of the input values, thus
    making the rounding approach for an FPTAS inapliccable. Luckily, there
    is an approach to bound the number of pairs in a more reasonable manner.

    The ``kabak`` implementation exloits a 2-factor approximation in conjunction
    with Lawler's [Law77]_ pair-based dynamic program. The primal-dual algorithm
    yields a linear-time 2-factor approximation for min cost knapsack [CS15]_.
    Let the apprixmate value be :math:`P_0` such that :math:`P_0 \leq  2C^*`.
    Now, constructing pairs starting with ``(0, 0)``, we can safely discard
    any pairs cost exceeds :math:`P_0`. At worst, this means maintaining
    :math:`2C^*` pairs. This yields a runtime comparable to that of the Knapsack
    algorithm. Moreover, this method admits input rounding for designing an
    FPTAS, as after rounding the value of  :math:`C^*` can be reduced. Indeed,
    using the bound :math:`P_0` makes the algorithms very closely resemble that
    of Lawler [Law77]_. The same methods can be used for constructing an optimal
    solution in linear time using the same backtracking tree strucutre.


    We can maintain the pairs in increasing order of cost. This way, while adding
    ``(cost[i], weight[i])`` to a pair, as soon as we find a pair ``(C, W)`` such
    that the sum ``C + cost[i]`` exceeds the upper bound we can break the loop and
    move to the next item. Moreover, the order is exploited at the end to find the
    pair with the lowest cost among all pairs with ``W >= demand``. This is simply
    the first pair that is feasible.


    """

    if sum(weight) < demand:
        return -1, []

    root = TreeNode(val=None, parent=None)

    pairs = [(0, 0, root)]

    for i, (c, w) in enumerate(zip(cost, weight)):
        newPairs = []

        for C, W, old_node in pairs:
            if C + c > upper_bound:
                break

            node = TreeNode(val=i, parent=old_node)

            newPairs.append((C + c, W + w, node))

        pairs = _merge_pairs(pairs, newPairs)

    # Find minimum costs pairs from sorted list
    for C, W, node in pairs:
        if W >= demand:
            val, _, current_node = C, W, node
            break

    if not return_sol:
        return val, []

    # construct solution
    sol = []

    while current_node.parent:
        sol.append(current_node.val)
        current_node = current_node.parent

    return val, sol


def _merge_pairs(oldPairs: list, newPairs: list) -> list:
    r"""
    Merge two sorted lists of ``(cost, weight, info)``-tuples.

    A ``(cost, weight)``-pair is said to be *dominated* if there is another
    pair with lower cost and more weight. If neither cost nor weight are
    strictly larger or smaller, respectively, preference is given to the
    ``old`` pair. This function merges two lists of sorted undominated pairs,
    while dropping any pairs that become dominated.

    The ``info`` entry allows us to carry additional information.


    Parameters
    ----------
    oldPairs : A list of ``(cost, weight, info)``-pairs in ascending order.
    newPairs : A list of ``(cost, weight, info)``-pairs in ascending order.

    Returns
    -------
    pairs : Sorted list of undominated ``(cost, weight, info)``-typles.
    """

    pairs = []

    i, j = 0, 0

    while (i < len(oldPairs)) and (j < len(newPairs)):
        c_old, w_old, *info_old = oldPairs[i]
        c_new, w_new, *info_new = newPairs[j]

        if (c_old <= c_new) and (w_old >= w_new):
            # New pair dominated by old
            pairs.append((c_old, w_old, *info_old))
            i += 1
            j += 1
            del info_new
            continue

        if (c_old >= c_new) and (w_old <= w_new):
            # Old pair dominated by new
            pairs.append((c_new, w_new, *info_new))
            i += 1
            j += 1

        if (c_old > c_new) and (w_old > w_new):
            # Incomparable, new lower cost
            pairs.append((c_new, w_new, *info_new))
            j += 1

        if (c_old < c_new) and (w_old < w_new):
            # Incomparable, old lower cost
            pairs.append((c_old, w_old, *info_old))
            i += 1

    # No new pairs remaining
    while i < len(oldPairs):
        pairs.append(oldPairs[i])
        i += 1

    # No old pairs remaining
    while j < len(newPairs):
        pairs.append(newPairs[j])
        j += 1

    return pairs


def _upper_bound(
    cost: ArrayLike, weight: ArrayLike, demand: float, method="primal-dual"
) -> tuple:
    """
    Find a constant-factor upper-bound for min cost Knapsack.

    Returns an upper-bound, lower-bound tuple.

    Parameters
    ----------
    cost : ArrayLike
    weight : ArrayLike
    demand : float
    method : str
        Method for finding constant factor approximations. Supports
        "primal-dual" and "greedy_half"
    """

    if method == "primal-dual":
        val, _ = primal_dual(cost, weight, demand)

    elif method == "greedy-half":
        val, _ = greedy_half(cost, weight, demand)

    else:
        raise ValueError(f"Method {method} not recoqnized. Try e.g `primal-dual`.")

    if val <= 0:
        return -1, -1

    return val, val / 2
