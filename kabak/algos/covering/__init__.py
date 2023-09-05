"""
Algorithms for Covering problems.

Introduction
============

Covering problems are quite general.
Therefore, many algorithms attain only moderately good worst-case performance;
the space of intances is simply so large that there are bound to be some
real bad eggs in it.
Nevertheless, there are some elegant approximation algorithms that attain
reasonable wors-case performance w.r.t. some parametrization of the instances.
In particular, the row and column sparsity of the constraint matrix ``A`` can be
exploited, as can the *width* of the instance based on the ratio of entries in
``A`` relative to ``b``. This module implements selectd sparsity-, and
width-exploiting algorithms.
"""

from kabak.algos.covering.grasp import grasp
from kabak.algos.covering.greedy import greedy
from kabak.algos.covering.primalDual import primal_dual
from kabak.algos.covering.strengthenedLinear import demand_values, solve_KCLP

__all__ = ["primal_dual", "solve_KCLP", "greedy", "grasp", "demand_values"]
