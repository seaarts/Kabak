from math import ceil
from multiprocessing import Pool

import numpy as np
from numpy.typing import ArrayLike

from kabak.algos.linearProgram.ortools import (
    _make_linear_program,
    _set_covering_constraints,
)
from kabak.algos.minKnapsack import greedy_half, rounding_fptas


def solve_KCLP(
    c: ArrayLike, A: ArrayLike, b: ArrayLike, eps: float, solver_type: int = 0
) -> tuple:
    r"""Solve linear programming relaxation of the problem with Google OR-tools.

    Approximate the KC-LP for a unit-muliplicity CIP.

    Parameters
    ----------
    c : ArrayLike
        Costs of each item.
    A : ArrayLike
        Matrix of covering constraints.
    b : ArrayLike
        Vector of covering demands.
    eps : float
        Tolerance for approximation error.
    solver_type: int
        Solver type passed to or-tools.

    Notes
    -----
    This algorithm is based on *Fast Algorithms for Solving the Knapsack-
    Cover LP* by Chandra Chekuri and Kent Quanrud [ChQu19]_. The main idea is
    based on the Multiplicative Weight Update (MWU) algorithm in the dual. We
    maintain weights :math:`w_i` for every item :math:`i \in F`, and KC-LP dual
    variables :math:`y^S_j` for every subset-user pair in :math:`2^F \times U`.
    Duals are upadted by solving a Lagrangian relaxation with the weights as
    penalties. This involves finding an approximalely *most violated inequality*.
    The weights are updates based on the degrees to which KC-LP dual cosntraints
    for items :math:`i \in F` are slack.

    While the main ideas derive from Carr *et al.* [Carr99]_, Chekuri and Quanrud
    propose a number of methods for speeding up computation, particularly lazy
    thresholding and and dynamic knapsack updates. Kabak implements these speedups,
    and applies parallelization where possible.
    """

    # Make initial linear program
    solver, x, cov_consts, _ = _make_linear_program(
        c, A, b, minimize=True, integral=False, solver_type=solver_type
    )

    # Make some ordering over users

    # While some user has violated ineq:
    #   Add ineq
    #   Resolve


def demand_values(demand: int, epsilon: float) -> list:
    r"""
    Generate KC-LP residual demand values to check for an epsilon violation.

    Notes
    -----
    Consider a minimum-cost knapsack problem with weights ``a`` and covering
    requirement ``demand``. The goal is to find the most violated KC-LP
    inequality, permitting an ``epsilon`` error. Assuming ``a`` and ``demand``
    are integral, it suffices to check a small number of potential residual
    demand values, rather than all possibilities. Letting :math:`D` denote
    the integer ``demand``, and assuming ``a <= demand``, the demadn values
    to check in order to find a :math:`\epsilon`-most violated inequality are:

    .. math::
        \mathcal{A} = \{\lceil (1 + \epsilon)^k \rceil: k = 0, 1, \dots,
        \lceil \log_{1 + \epsilon}(\max_i \{D\})\rceil \}

    """

    n_vals = max(int(np.ceil(np.log(demand) / np.log(1 + epsilon))), 0) + 1

    return list(set(ceil((1 + epsilon) ** k) for k in range(n_vals)))


def _violated_inequality(
    user: int,
    x: ArrayLike,
    A: ArrayLike,
    b: ArrayLike,
    epsilon: float,
    tol: float,
    nCPU=4,
) -> tuple:
    """
    Find most violated KC-inequality of given user via min-cost knapsack.

    Parameters
    ----------
    nCPU : int
        Number of parallel processes used for solving MC-KC.

    Notes
    -----
    This algorithms generates and solves a sequence of min-cost knapask problems.

    The algorithms returns the set of items defining the constraint, the residual
    contributions, and the residual demand.

    The item set can be frozen and hashed to ensure the same constraint is not
    re-discovered in a subsequent iteration for debugging.
    """

    a_int, demand_int = np.ceil(A[user, :] / tol), np.ceil(b[user] / tol)

    d_values = demand_values(demand_int, epsilon)

    def _violation(d):
        return _max_violation(a_int, demand_int, x, d, epsilon)

    # solve with multiprocessing
    pool = Pool()

    results = pool.map(_violation, d_values)

    return results


def _max_violation(a, demand, x, d, epsilon):
    """
    Find the most violated KC-LP inequality for a given user.

    Parameters
    ----------
    a : ArrayLike
        A row vector of integral min-cost knapsack item contributions.
    demand : int
        Integral demand of min-cost knapsack.
    d : float
        Target residual demand.
    x : ArrayLike
        A fractional knapsack solution.
    epsilon : float
        A small error tolerance.
    """

    a_residual = np.minimum(a, d)

    cost = a_residual * x / d

    requirement = a.sum() - demand + d

    return rounding_fptas(cost, a_residual, requirement, eps=epsilon, return_sol=True)
