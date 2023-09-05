r"""
An implementation of the dynamic programming for ``{0,1}``-Knapsack.

See Lawler, Eugene L.  "Fast Approximation Algorithms for Knapsack Problems.",
*18th Annual Symposium on Fundamentals of Computer Science (SFCS 1977).* IEEE,
1977.

"""

from numpy.typing import ArrayLike

from kabak.structures import TreeNode


def _merge_pairs(oldPairs: list, newPairs: list) -> list:
    r"""
    Merge two sorted lists of ``(profit, weight, info)``-tuples.

    A ``(profit, weight)``-pair is said to be *dominated* if there is another
    pair with more profit and less weight. If neither profit nor weight are
    strictly larger or smaller, respectively, preference is given to the
    ``old`` pair. This function merges two lists of sorted undominated pairs,
    while dropping any pairs that become dominated.

    The ``info`` entry allows us to carry additional information.

    Parameters
    ----------
    oldPairs : A list of ``(profit, weight, info)``-pairs in ascending order.
    newPairs : A list of ``(profit, weight, info)``-pairs in ascending order.

    Returns
    -------
    pairs : Sorted list of undominated ``(profit, weight, info)``-pairs.
    """

    pairs = []

    i, j = 0, 0

    while (i < len(oldPairs)) and (j < len(newPairs)):
        p_old, w_old, *info_old = oldPairs[i]
        p_new, w_new, *info_new = newPairs[j]

        if (p_old >= p_new) and (w_old <= w_new):
            pairs.append((p_old, w_old, *info_old))
            i += 1
            j += 1
            del info_new
            continue

        if (p_old <= p_new) and (w_old >= w_new):
            pairs.append((p_new, w_new, *info_new))
            i += 1
            j += 1

        if (p_old < p_new) and (w_old < w_new):
            pairs.append((p_old, w_old, *info_old))
            i += 1

        if (p_old > p_new) and (w_old > w_new):
            pairs.append((p_new, w_new, *info_new))
            j += 1

    # No new pairs remaining
    while i < len(oldPairs):
        pairs.append(oldPairs[i])
        i += 1

    # No old pairs remaining
    while j < len(newPairs):
        pairs.append(newPairs[j])
        j += 1

    return pairs


def optimal_value(profit: ArrayLike, weight: ArrayLike, budget: int) -> int:
    r"""
    Return the maximum value of Knapsack packing.

    The runtime is :math:`\mathcal{O}(n\min\{P^*, b\})`, where :math:`P^*` is
    the maximum profit and :math:`b` is the ``budget``. This is a pseudo-
    polynomial time algorithm.

    Parameters
    ----------
    profit : Positive integer profits of items, shape ``(n_items,)``.
    weight : Positive integer weights of items, shape ``(n_imtes, )``.
    budget : Positive integer budget.
    """

    pairs = [(0, 0)]

    for p, w in zip(profit, weight):
        newPairs = []

        for p_old, w_old in pairs:
            if w_old + w > budget:
                break

            newPairs.append((p_old + p, w_old + w))

        pairs = _merge_pairs(pairs, newPairs)

    return pairs[-1][0]


def optimal_solution(profit: ArrayLike, weight: ArrayLike, budget: int) -> tuple:
    r"""
    Return the maximum value of Knapsack, and a list of item indices attaining
    this value.

    Parameters
    ----------
    profit : Positive integer profits of items, shape ``(n_items,)``.
    weight : Positive integer weights of items, shape ``(n_imtes, )``.
    budget : Positive integer budget.

    Notes
    -----
    This implementation uses a rooted tree of size
    :math:`\mathcal{O}(n\min\{P^*, b\})` which is backtracked to retreive
    the optimal solution. Each profit-weight tuple ``(p, w)`` also carries
    a pointer to the node in the rooted tree. Backtracking via ``node.parent``
    until the ``root``, collecting  ``node.val``-values along the path yields
    the indices of a solution.
    """

    root = TreeNode(val=None, parent=None)

    pairs = [(0, 0, root)]

    for i, (p, w) in enumerate(zip(profit, weight)):
        newPairs = []

        for p_old, w_old, node_old in pairs:
            if w_old + w > budget:
                break

            node = TreeNode(val=i, parent=node_old)

            newPairs.append((p_old + p, w_old + w, node))

        pairs = _merge_pairs(pairs, newPairs)

    opt_val = pairs[-1][0]

    # Fetch optimum value and solution
    opt_items = []

    current_node = pairs[-1][2]

    while current_node.parent:
        opt_items.append(current_node.val)
        current_node = current_node.parent

    return opt_val, opt_items
