import numpy as np
from numpy.typing import ArrayLike


def _greedy_update(
    A_resid: ArrayLike, r_resid: ArrayLike, f_fixed: ArrayLike, unbuilt: set
) -> tuple:
    """
    Finds the highest value facility.

    Notes
    -----
    (Taken from loraplan - may need streamlining).

    This algorithm implements a greedy step update for Covering Integer
    Programs (CIPs). It takes a vector of resiudal demands ``r_resid``, a
    matrix of residual contributions ``A_resid``, a fixed cost vector
    ``f_fixed``, and a set of indices of unbuilt facilities. It returns the
    index of an index that minimizes the cost per residual coverage, as well
    as updated residual values, contrbutions, and unbilt indices.

    Parameters
    ----------
    r_resid : ArrayLike
        (nDems, ) vector of residual requirements for each demand.
    A_resid : ArrayLike
        (nDems, nFacs) matrix of residual contributions.
    f_fixed : ArrayLike
        (n_facs, ) vector of fixed facility costs.
    unbuild : set
        Set of indices of unbuild facilities.

    Returns
    -------
    reqs_new : ArrayLike
        (nDems, ) vector of updated residual requirements.
    A_new : ArrayLike
        (nDems, nFacs) matrix of updated residual contribtions.
    unbuild_new : set
        Updated set of unbuilt indices.
    facility : int
        Index of selected facility.
    """

    nDems = len(r_resid)
    alive = np.where(r_resid > 0)[0]  # alive users
    unconst = np.array(list(unbuilt))  # unconstructed facilities

    # get relevant sub-arrays
    A = A_resid[np.ix_(alive, unconst)]
    f = f_fixed[np.ix_(unconst)]

    # compute contributions by as column sums
    contributions = A.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        unit_costs = f / contributions  # unit-cost of contributions

    # select minimum costs facility
    best_fac = np.argmin(unit_costs)  # select lowest cost facility
    facility = unconst[best_fac]  # global index of selected facility

    # update residual instance
    reqs_new = np.maximum(r_resid - A_resid[:, facility], 0)
    a_max = reqs_new.reshape(nDems, 1)  # make broadcastable with A
    A_new = np.minimum(A_resid, a_max)
    unbuilt_new = unbuilt - set([facility])

    return reqs_new, A_new, unbuilt_new, facility


def greedy(A: ArrayLike, r: ArrayLike, f: ArrayLike, eval_factor: bool = False) -> dict:
    """
    Run greedy algorithm for CIPs.

    Parameters
    ----------
    r : ArrayLike
        (nDems, ) vector of requirements for each demand.
    A : ArrayLike
        (nDems, nFacs) matrix of contributions.
    f : ArrayLike
        (n_facs, ) vector of fixed facility costs.
    eval_factor : bool
        Set to ``True`` to evaluate APX-factor.

    Returns
    -------
    out : dict
        Output dictionary containing ``cost`` and constructed ``facilities``.

    Notes
    -----
    The ``greedy`` algorithim is a classic iterative algorithms for solving hard
    problems. See Dobson [Dob82]_ for reference and analysis on covering programs.

    If the inputs are integral, this is a
    :math:`\mathcal{O}(\log(n))`-approximation algorithm.
    """

    nDems, nFacs = len(r), len(f)

    if A.shape[0] != nDems:
        raise ValueError("Dimension 0 of `A` and len of `r` do not match.")
    elif A.shape[1] != nFacs:
        raise ValueError("Dimension 1 of `A` and len of `f` do not match.")

    # residual values
    r_res = r
    A_res = A
    unbuilt = set(np.arange(len(f)))

    # shave off excess contributions of A
    a_max = r_res.reshape(nDems, 1)
    A_res = np.minimum(A_res, a_max)

    # standardize instance
    A_res = A_res / r_res[:, np.newaxis]  # A_res in [0,1]
    r_res = np.repeat(1, r_res.shape)

    construct = []
    residuals = []

    # store old columnsum for optional factor evaluation
    oldcolsum = A_res.sum(axis=0)
    factors = []

    # main loop
    while np.any(r_res > 0):
        if not bool(unbuilt):  # if no unbuilt facilities
            return {
                "cost": np.NaN,
                "sol": construct,
                "residuals": residuals,
                "factor": sum(factors).max(),
                "feasible": False,
            }
        else:
            r_res, A_res, unbuilt, fac = _greedy_update(A_res, r_res, f, unbuilt)
            construct.append(fac)
            residuals.append(r_res)

            if eval_factor:
                # compute new column sum and factors
                newcolsum = A_res.sum(axis=0)

                with np.errstate(divide="ignore", invalid="ignore"):
                    factor = (oldcolsum - newcolsum) / oldcolsum

                # replace NaNs (0/0) and Infs (a/0, if a != 0) with 0s
                factor[np.isnan(factor)] = 0
                factor[np.isinf(factor)] = 0

                factors.append(factor)
                oldcolsum = newcolsum

    # compute cost
    cost = 0
    for fac in construct:
        cost = cost + f[fac]

    if eval_factor:
        # compute max factor and return
        max_fac = sum(factors).max()
        return {
            "cost": cost,
            "sol": construct,
            "residuals": residuals,
            "factor": max_fac,
            "feasible": True,
        }
    else:
        return {"cost": cost, "sol": construct, "feasible": True}
