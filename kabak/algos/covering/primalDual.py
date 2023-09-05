import numpy as np
import numpy.ma as ma
from numpy.typing import ArrayLike


def primal_dual(
    A: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
):
    r"""
    A primal dual algorithm for covering integer programs.

    Parameters
    ==========
    A : ArrayLike
        A matrix of non-negative contributions of shape ``(n_dem, n_var)``.
    b : ArrayLike
        A vector of positive demands of shape ``(n_dem, )``
    c : ArrayLike
        A vector of item costs of shape ``(n_var,)``.
    return_sol : bool
        Indicates whether or not to return the solution along the solution value.
    return_dauls: bool:
        Indicates whether or not to return the dual solution.
    """

    nDems, nFacs = len(b), len(c)

    if A.shape[0] != nDems:
        raise ValueError("Dimension 0 of A and len of r do not match.")
    if A.shape[1] != nFacs:
        raise ValueError("Dimension 1 of A and lenr of f do not match.")

    # initialize residual values
    A_res, b_res, c_res = A, b, c
    unbuilt = set(np.arange(nFacs))

    # shave off excess contributions of A
    a_max = b_res.reshape(nDems, 1)
    A_res = np.minimum(A_res, a_max)

    construct, residuals = [], []
    residuals.append(b)

    # main loop
    while np.any(b_res > 0):
        if not bool(unbuilt):  # if no unbuilt facilities
            return {"cost": np.NaN, "sol": construct, "feasible": False}
        else:
            # fetch next facility
            A_res, b_res, c_res, unbuilt, fac = _dual_update(
                A_res, b_res, c_res, unbuilt
            )
            # update outputs
            construct.append(fac)
            residuals.append(b_res)

    # compute cost
    cost = c[construct].sum()

    return {
        "cost": cost,
        "facilities": construct,
        "residuals": residuals,
        "feasible": True,
    }


def _dual_update(A_resid, b_resid, c_resid, unbuilt):
    """
    A dual update for the primal-dual algorithm for CIPs.
    Choose the next facility to construct out of a set
    of previously unbuilt facilities, and updates residual
    values for requirements r, contributions A, and costs f.

    Prameters
    ---------
    A_resid : array_like
        A ``(nDems, nFacs)`` array of contributions.
    b_resid : array_like
        A ``(nDems, )`` array of residual requirements.
    f_resid : array_like
        A ``(nFacs)`` array of residual costs.
    unbuilt : set
        Available indices of unbuilt facilities.
    """

    n_dems = len(b_resid)
    livings = np.where(b_resid > 0)[0]  # alive users
    unconst = np.array(list(unbuilt))  # unconstructed facilities

    # get relevant sub-arrays
    A = A_resid[np.ix_(livings, unconst)]
    c = c_resid[np.ix_(unconst)]

    contributions = A.sum(axis=0)  # contribution of each facility
    with np.errstate(divide="ignore", invalid="ignore"):
        unit_costs = c / contributions

    # compute minimum costs and select best facility
    min_cost = np.min(unit_costs)  # find lowest unit cost
    best_fac = np.argmin(unit_costs)  # select lowest cost facility
    facility = unconst[best_fac]  # global index of selected facility

    # residual requirements
    b_new = np.maximum(b_resid - A_resid[:, facility], 0)

    a_max = b_new.reshape(n_dems, 1)  # make broadcastable with A
    A_new = np.minimum(A_resid, a_max)  # take min over A and residual requirements
    c_new = c_resid - A_resid.sum(axis=0) * min_cost  # adjust remaining prices
    # note: dead users have A_resid == 0 and so do not contribute

    unbuilt_new = unbuilt - set([facility])

    return A_new, b_new, c_new, unbuilt_new, facility
