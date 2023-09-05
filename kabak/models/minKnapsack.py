import warnings

import numpy as np
from numpy.typing import ArrayLike

from kabak.models.covering import CoveringModel


def _check_inputs(cost: ArrayLike, weight: ArrayLike, budget: int) -> tuple:
    """Verify inputs are valid."""
    cost = np.array(cost)
    weight = np.array(weight)

    if cost.shape != weight.shape:
        raise ValueError(
            f"Shape of cost and weight not matching ({cost.shape} {weight.shape})."
        )

    if not budget > 0:
        raise ValueError("Budget must be positive.")

    if not all(cost > 0):
        raise ValueError("All costs must be strictly positive.")

    if not all(weight > 0):
        raise ValueError("All weights must be strictly positive.")

    if sum(weight) < budget:
        warnings.warn(
            "The instance is infeasible. Try increasing the budget", UserWarning
        )

    return cost, weight, budget


class MinKnapsackModel(CoveringModel):
    r"""A Minimum-Cost Knapsack model - inerits from ``CoveringModel``.


    The base class has three main components.

    - An objective ``costs`` to minimize.
    - A minimum weight constraint specified via ``weights, budget`` such that
      ``weights * x >= budget`` (where ``*`` denotes a vector product).
    - An optional multiplicity constraint  ``d``, such that ``x <= d``. The
      default is ``d = 1``.


    Model formulation
    *****************

    Mathmatically the minimum cost knapsack problem is the following:

    .. math::

      \min_x c^T &x \\
      \text{s.t. } w^T &x \geq B \\
      &x \leq  d\\
      &x \geq 0

    - The vector :math:`c` are the ``cost``-entries.
    - The vecotor :math:`x` is the decison variable, :math:`x_i` indicating
      whether or not to buy item :math:`i`.
    - The vector :math:`w` are item ``weights``
    - The covering constraint :math:`w^Tx \geq B` means a subset of items of
      weight at least :math:`B` must be selected.
    - Here :math:`B` is a demand (or minimum ``budget``).
    - The vector :math:`d` are multiplicity constraints on how many copies of
      each item may be taken. Typically these are ``1``.
    - All entries are assumed to be non-negative.

    Integrality Gap
    ***************

    The Min Cost Knapsack problem has an unbounded integrality gap,
    just like the Covering Integer Program.

    Knapsack-cover Inequalities
    ***************************
    
    Many approximation algorithms make use of knapsack-cover inequalities.
      
    
    Solvability
    ***********

    The Minimum Knapsack problem is relatively easy to solve despite being NP-hard.
    When all inputs are integral, the problem admits both exact pseudo-polynomial
    time solutions via dynamic programming, as well as a fully polynoial-time
    approximation scheme (FPTAS) based on input rounding.

    If the input data are fractional, it must be rounded in order to use the DP.
    This is not a problem if employing the FPTAS. We specify a precision
    :math:`K > 0` and use scaled costs as :math:`c'_i = K \lfloor c_i / K \rfloor`.
    The loss from rounding is bounded using the same analysis as for the FPTAS.

    """

    def __init__(self, cost: ArrayLike, weight: ArrayLike, budget: int):
        """
        Initialize model and verify inputs.
        """

        cost, weight, budget = _check_inputs(cost, weight, budget)

        self.cost = cost
        self.weight = weight
        self.budget = budget

        self.integral_input = all(c.is_integer() for c in cost) and all(
            w.is_integer() for w in weight
        )

        super().init(self, c=cost, A=self.weight, b=self.budget)
