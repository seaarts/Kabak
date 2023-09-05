r"""
Algorithms for the min cost knapsack problem.


Introduction
============

The minimum knapsack problem is very similar to the
classic knapsack problem, however requires some aglorithms designed unquely for
minimum cost knapsack problems.
In particualar, while exact algorithms for Knapsack can 
trivially be converted into algorithm for min cost knapsack, the same is not true
for approximation algorithms. Carr *et al*. [Carr99]_ state that:

    While the knapsack problem and the minimum knapsack problems are equivalent
    if an exact solution is sought, they are not equivalent for approximation
    purposes in that a :math:`\rho`-approximation algorithm for one problem
    does not imply the existence of a comparable guarantee for the second.

As such, ``kabak`` implements a tailor-made FPTAS for min cost knapsack, while
``models.minKnapsack`` uses the DP-algorithm of ``models.knapsack`` for its
exact ``solve_exact()``-method.


Below are the available algorithms and their implementations.

"""

from kabak.algos.minKnapsack.dynamic_program import dynamic_program_bounded
from kabak.algos.minKnapsack.greedy import greedy_half
from kabak.algos.minKnapsack.primal_dual import primal_dual
from kabak.algos.minKnapsack.rounding import rounding_fptas

__all__ = [
    "dynamic_program_bounded",
    "rounding_fptas",
    "primal_dual",
    "greedy_half",
]
