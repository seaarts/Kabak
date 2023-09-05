import numpy as np
from numpy.typing import ArrayLike

from kabak.algos.knapsack import (
    optimal_solution,
    optimal_value,
    rounding_fptas,
    solve_relaxation,
)
from kabak.models.packing import PackingModel


class KnapsackModel(PackingModel):
    r"""A Knapsack model class - inerits from ``PackingModel``.

    The base class has three main components.

    - An objective ``profit`` to maximize.
    - A maximum weight constraint specified via ``weights, budget`` such that
      ``weights * x <= budget`` (where ``*`` denotes a vector product).
    - An optional multiplicity constraint  ``d``, such that ``x <= d``. The
      default is ``d = 1``.


    Model formulation
    *****************

    Mathmatically the :math:`\{0, 1\}`-knapsack problem is the following:

    .. math::

      \max_x p^T &x \\
      \text{s.t. } w^T &x \leq B \\
      &x \leq  1\\
      &x \geq 0

    - The vector :math:`p` are the item ``profit``-values.
    - The vecotor :math:`x` is the decison variable, :math:`x_i` indicating
      whether or not to buy item :math:`i`.
    - The vector :math:`w` are item ``weights``
    - The budget constraint :math:`w^Tx \leq B` restructs selections to have weight at most :math:`B`.
    - All entries are assumed to be non-negative.

    """

    def __init__(self, profit: ArrayLike, weight: ArrayLike, budget: int):
        self.cost = np.array(profit)
        self.weight = np.array(weight)
        self.budget = budget

        self.integral_input = all(p.is_integer() for p in profit) and all(
            w.is_integer() for w in weight
        )

        super().init(
            self, profit=profit, B=self.weight, f=self.budget, d=np.ones(len(profit))
        )

    def solve_exact(self, return_sol: bool = False):
        r"""
        Find an exact integer solution to Knapsack.

        Parameters
        ----------
        return_sol : Bool
            Indiactes whether or not to return the optimal solution in addition
            to the optimal value.

        Notes
        -----
        The ``knapsack`` model admits an exact solution algorithm that runs in
        `pseudo-polynomial time
        <https://en.wikipedia.org/wiki/Pseudo-polynomial_time>`_
        :math:`\mathcal{O}(n\min\{P^*, b\})` where :math:`P^*` is the maximum
        profit, and :math:`b` is the budget.

        The algorithms is founded on dynamic programming.
        """
        if return_sol:
            return optimal_solution(self.profit, self.weight, self.budget)

        return optimal_value(self.profit, self.weight, self.budget)

    def solve_fractional(self, return_sol: bool = False):
        """
        Solve the linear programming relaxation of Knapsack.

        Parameters
        ----------
        return_sol : (optional) If ``True`` the solution is also returned.

        Notes
        -----
        This uses a specialized algorithms based on sorting.
        The items are sorted in decreasing
        order of ``profit / weight`` and taken (wholly or fractionally) until
        the budget is exhausted. This has time-complexity :math:`\mathcal{O}(
        n\log n)`.

        .. todo::
          Implement the (theoretically) faster algorithm based on median-finding. (See [Law77]_).

        """

        return solve_relaxation(
            self.profit, self.weight, self.budget, return_sol=return_sol
        )

    def approximate(self, eps: float = 0.01, return_sol: bool = False):
        r"""
        Return a solution with value at least ``(1 - eps)`` times the optimum.

        Parameters
        ----------
        eps : The approximation factor; smaller values yield better sloutions
           at the cost of computational load.
        return_sol : Whether to return the solution (in addition to the value).


        Notes
        -----

        This implementation is based on *input rounding*. This is implemented
        following [Law77]_. The rough idea is to divide profits ``p`` by some
        value ``K`` so that the runtime is a function of ``P_max/K`` rather than
        ``P_max``. A careful choice of rounding factor ``K`` balances solution
        quality versus runtime.

        Suppose all profits :math:`p_i` are divided by some factor :math:`K > 1`,
        and rounded down. Let :math:`q_i = \lfloor p_i / K \rfloor`. Clearly the
        runtime is reduced from :math:`\mathcal{O}(n \min\{P^*, b\})` to
        :math:`\mathcal{O}(n \min\{P^*/K, b\})`. The solution quality is also
        reduced, however the loss in quality can be bounded. For any item

        .. math::

          K q_j \leq p_j < K(q_j + 1).

        Hence, for any set :math:`S` of items, the loss is bounded by

        .. math::

          \sum_{i \in S} p_i - K \sum_{i\in S }q_i < K |S| \leq Kn

        To have a :math:`(1 - \epsilon)`-approximation we would like to bound
        the error by a funciton of :math:`\epsilon` and the optimal value
        :math:`P^*`. In particular, we wish to establish that

        .. math::

          Kn \leq \epsilon P^*

        This shows it suffices to ensure that :math:`K \leq \epsilon P^* / n`.
        Because we do not know the optimum value :math:`P^*` we employ an
        upper-bound in its place. This makes `K` larger, and slows the algorithm
        down, so the tighter the bound the better. Our implementation uses the
        a greedy approximation :meth:`kabak.algos.knapsack.linear.greedy_approx`.
        This returns an approximately optimal value :math:`P_0`
        that satisfies :math:`P^* \leq 2 P_0`.
        This translates to a rounding factor :math:`K = 2P_0\epsilon / n`.
        The implied runtime is :math:`\mathcal{O}(n\min\{n/\epsilon, b\})`.

        """
        if eps >= 1:
            raise Warning(
                ValueError,
                "An epsilon >= 1.0 permits arbitrarily bad solutions.\
                          Try using a smaller value.",
            )

        val, sol = rounding_fptas(self.profit, self.weight, self.budget, eps=eps)

        if return_sol:
            return val, sol

        return sol
