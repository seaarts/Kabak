import numpy as np
import numpy.ma as ma
from numpy.typing import ArrayLike


def primal_dual(
    cost: ArrayLike,
    weight: ArrayLike,
    demand: float,
    return_sol=False,
    return_duals=False,
) -> tuple:
    """
    Find and approximately minimum cost feasible knapsack.

    Parameters
    ----------
    cost : ArrayLike
        Vector of positive item costs.
    weight : ArrayLike
        Vector of positive item weights.
    demand : float
        The minimum amount of weight to be packed.
    return_sol : bool
        Whether to return the solution in addition to the approximately
        optimal value.
    return_dual : bool
        Whether to return the feasible dual solution constructed.

    Returns
    -------
    (val, sol) : tuple
        A tuple containing the value and solution, respectively.

    Notes
    -----
    This is a primal-dual algorithm based on Carnes and Shmoys [CS15]_.
    The algorithm starts with an infeasible primal solution, and a feasible
    dual solution, both having all variables set to zero. Then, the dual
    variables are uniformly increased until some constraint becomes tight.
    The item corresponding to the tight constraint is bought. One the
    residual demand is zero, the algorithm terminates with a feasible
    approximately optimal integral primal solution, and a feasible fractional
    dual solution.
    """

    if (len(cost) == 0 or len(weight) == 0) and return_duals:
        return 0, [], []

    if len(cost) == 0 or len(weight) == 0:
        return 0, []

    duals = []

    fill_value = -99999

    # cost.copy() is necessary in case cost is a numpy array.
    amortized_cost = ma.masked_array(
        cost.copy(), mask=False, fill_value=fill_value, dtype=np.float64
    )

    remaining_weight = ma.masked_array(
        weight, mask=False, fill_value=fill_value, dtype=np.float64
    )

    residual_demand = demand

    selected = np.zeros(len(cost))

    for _ in range(len(cost)):
        if residual_demand <= 0:
            break

        # Run primal dual update
        remaining_weight = np.minimum(remaining_weight, residual_demand)

        unit_cost = amortized_cost / remaining_weight

        item = np.argmin(unit_cost)  # select min cost item

        selected[item] = 1
        residual_demand -= remaining_weight[item]

        # Update amortized costs
        dual = unit_cost[item]
        amortized_cost -= remaining_weight * dual

        if return_duals:
            duals.append(dual)

        # Update masks
        remaining_weight.mask = selected
        amortized_cost.mask = selected

    if residual_demand > 0 and return_duals:
        return -1, [], []

    if residual_demand > 0:
        return -1, []

    val = np.dot(cost, selected)

    if not return_sol and not return_duals:
        return val, []

    sol = np.where(selected == 1)[0].tolist()

    if not return_duals:
        return val, sol

    return val, sol, duals
