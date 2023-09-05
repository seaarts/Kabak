import numpy as np
from numpy.typing import ArrayLike


def _contributions(
    A_resid: ArrayLike, b_resid: ArrayLike, c_fixed: ArrayLike, unbuilt: set
) -> tuple:
    """
    Get list of (unit-cost, index)-pairs of unbilt facilities in ascending order.
    """

    alive = np.where(b_resid > 0)[0]
    unbuilt = np.array(list(unbuilt))

    # subset arrays
    A = A_resid[np.ix_(alive, unbuilt)]
    c = c_fixed[np.ix_(unbuilt)]

    contributions = A.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        unit_costs = c / contributions

    # collecct (unit_cost, facility_id) pairs of positive-contribution items
    out = [
        (cost, unbuilt[i]) for i, cost in enumerate(unit_costs) if cost < float("inf")
    ]

    return sorted(out)


def _local_search_eliminate(A: ArrayLike, b: ArrayLike, built=list):
    """Returns solution with redundant facilities removed.

    Processes failities in reverse order.
    """

    excess = A[:, built].sum(axis=1) - b

    retained_facs = []

    for fac in reversed(built):
        if np.all(excess - A[:, fac] >= 0):
            excess -= A[:, fac]
        else:
            retained_facs.append(fac)

    return retained_facs


def grasp(
    A: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    minvalue: float = 0.8,
    maxsize: int = None,
    seed: int = None,
) -> dict:
    """
    Returns randomized heuristic solution using GRASP.

    Notes
    -----
    GRASP is a meta-heuristic called a Greedy Randomized Adaptive Search
    Procedure. Feo and Resende [FeRe95]_ is a standard reference. The procdure
    iteratively makes random selections from a restricted candidate list to
    build a feasible solution. Before termination a local search procedure
    attempts to improve on the random solution. There is some variation in the
    procedure depeding on how the restricted candidate lists are created.

    The restricted candidate lists are subsets of all available items. In our
    implementation items are sorted in ascending order of per-unit cost. We
    can select any items within a fraction ``alpha < 1`` of the best item, and
    / or select the best ``beta`` items in the list. If we allow all items
    this is a random walk, of only one time is premitted this is the greedy
    algorithm, and there is no randomness. Usually some variance makes it more
    likely to hit a global optimum.
    """

    rng = np.random.default_rng(seed=seed)

    nDems, nFacs = len(b), len(c)

    if A.shape[0] != nDems:
        raise ValueError("Dimension 0 of `A` and len of `r` do not match.")

    if A.shape[1] != nFacs:
        raise ValueError("Dimension 1 of `A` and len of `f` do not match.")

    if not maxsize:
        maxsize = nFacs

    # residual values
    A_res, b_res = A, b
    unbuilt = set(range(nFacs))

    # shave off excess contributions of A
    A_res = np.minimum(A_res, b_res.reshape(nDems, 1))

    # standardize instance
    A_res = A_res / b_res[:, np.newaxis]
    b_res = np.repeat(1, nDems)

    built = []

    # main loop
    while np.any(b_res > 0):
        if not unbuilt:
            return {"cost": np.NaN, "sol": built, "status": "infeasible"}

        unit_costs = _contributions(A_res, b_res, c, unbuilt)

        # collect restricted candidate list
        candidates = []
        for n, (cost, fac_id) in enumerate(unit_costs):
            if n >= maxsize or minvalue * cost > unit_costs[0][0]:
                break

            candidates.append(fac_id)

        # select facility
        facility = rng.choice(candidates)
        built.append(facility)

        # update residual instance
        b_res = np.maximum(b_res - A_res[:, facility], 0)
        A_res = np.minimum(A_res, b_res.reshape(nDems, 1))
        unbuilt -= set([facility])

    # Local-search
    built = _local_search_eliminate(A, b, built)

    # collect value and solution
    val = c[built].sum()
    sol = [int(i in built) for i in range(nFacs)]

    return {"cost": val, "sol": sol, "status": "feasible"}
