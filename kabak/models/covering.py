import numpy as np

from kabak.models.base import BaseModel


class CoveringModel(BaseModel):
    r"""A Covering model class - inerits from ``BaseModel``.

  
    The base class has three main components.

    - An objective ``costs`` to minimize.
    - Covering constraints specified via ``A, b`` such that ``Ax >= b``.
    - An optional multiplicity constraint  ``d``, such that ``x <= d``.

    Mathmatically the covering problem is the follwing:

    .. math::

        \min_x c^T &x \\
        \text{s.t. } A&x \geq b \\
        &x \leq  d\\
        &x \geq 0

    Where :math:`c` are the ``cost``-entries, :math:`(A, b)` specify the covering
    constraints, and :math:`d` specifies the multiplicity constriants.
    All entries are assumed to be non-negative.
    
    The ``Model`` does  **not** specify whether decision variables ``x`` are
    integral or fractions, because it is often relevant to find both fractional
    and integral solutions for the same model.


    Integrality Gap
    ***************

    The  Covering Integer Program. problem has an unbounded integrality gap [Carr99]_.

    Knapsack-cover Inequalities
    ***************************
    
    Many approximation algorithms make use of Knapsack-cover inequalities.

        
    """

    def __init__(self, c, A, b, d=None):
        super().init(c, A, b, B=None, f=None, d=d, kind="minimize")

    def contributions(self, selected):
        """
        The contribution of each unbuilt item towards satisfying the constratins.

        Fix a selection ``S = selected`` of items.
        For each ``A[i,j]`` in ``A`` let ``A[S][i,j] = min(A[i,j], r[S][j])``,
        where ``r[S][j]`` is the *residual demand* of constraint ``j`` after
        having selected items ``S``. Theese residual demands may be zero.

        The *contribution* of item ``i`` is ``sum(A[S][i,j] for j in constraints)``,
        where ``constraints`` is the set of covering constraints encoded in ``A``.
        """
