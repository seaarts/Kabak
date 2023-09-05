import numpy as np


class BaseModel:
    r"""A Covering / Packing model base class.

    Notes
    -----
    The base class has three main components.

    - An **ojective** specifying the ``kind`` (``"maximize"`` or ``"minimize"``)
      and the ``costs``.
    - **Packing / covering constraints** specified via matrices and vectors ``A, b, B, f``.
    - **Multiplicity constraints** indicating the maximum number of times items
      can be selected, specified by ``d``.

    An example of a a generic **covering** problem (``kind="minimize"``) is:

    .. math::

        \min_x c^T &x \\
        \text{s.t. } A&x \geq b \\
        B&x \leq f \\
        &x \leq  d\\
        &x \geq 0

    Where :math:`c` are the `cost`-entries, :math:`(A, b)` specify the covering
    constraints, :math:`(B, f)` specify the packing constraints, and
    :math:`d` specifies the multiplicity constriants.  All entries are assumed
    to be non-negative. An analagous **packing** problem aims to maximize the
    objective, instead.
    

    .. note::

      The ``BaseModel`` does  **not** specify whether decision variables
      ``x`` are integral or fractional. It is often relevant to find both
      fractional and integral solutions for the same model. Integrality of
      the output is specified when calling ``solve()`` or ``approximate()``.

      However, some models may have a ``selected``-attribute. This attribute
      may be used by ``solve`` or ``approximate`` algorithms. These may exploit
      specialized methods for updating the ``contributions`` or other attrubutes,
      in which case maintaining an internal ``self.selected``-attribute can be
      useful. The ``selected`` attribute may change between integral and fractional
      depending on the solution method using it. It should not be stored between
      calls to different solution methods.

    """

    def __init__(self, c, A, b, B, f, d, kind="minimize"):
        """
          Initialize BaseModel class.


        Parameters
        ----------
        c : ndarray
          Vector of item costs of size ``(n_items,)``
        A : ndarray
          Matrix of covering contributions of shape ``(n_cov, n_items)``
        b : ndarray
          Vector covering requirements of shape ``(n_cov,)``
        B : ndarray
          Matrix of packing contributions of shape ``(n_pack, n_items)``
        f : ndarray
          Vector of packing requirements of shape ``(n_pack,)``
        d : ndarray
          Vector of multiplicity constraints of shape ``(n_items)``
        kind : str
          The kind of optimization (``"minimize"`` or ``"maximize"``)
        """
        if kind not in {"minimize", "maximize"}:
            raise ValueError(f"kind must be `minimize` or `maximize`, not {kind}.")

        self.c = c
        self.A = A
        self.b = b
        self.B = B
        self.f = f
        self.d = d

        self.kind = kind

    def solve(self):
        """
        **(Placeholder)** Solve the model exactly.

        Sub-classes that do not implement ``solve()`` are permitted.
        Some models may only admit appximation algorithms or heuristics in
        in reasonable time.

        Override this for sub-models that admit useful exact algorithms.
        """
        raise NotImplementedError("``solve()`` is not implement for this model.")

    def approximate(self):
        """
        **(Placeholder)** Find an approximately optimal solution.

        Sub-classes that do not implement ``approximate()`` are permitted.
        Some models may only admit heuristics in reasonable time, or may
        admit efficient exact algorithms, negating the need for approxmiation.

        Override this for sub-models that admit useful approximation algorithms.
        """

        raise NotImplementedError("``approximate()`` is not implement for this model.")
