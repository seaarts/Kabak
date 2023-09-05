import numpy as np

from kabak.models.base import BaseModel


class PackingModel(BaseModel):
    r"""A Packing model class - inerits from ``BaseModel``.

  
    The packing model has three main components.

    - An objective of ``profits`` to maximize.
    - Covering constraints specified via ``B, f`` such that ``Bx <= f``.
    - Optional multiplicity constraints  ``d``, such that ``x <= d``.

    Mathmatically the packing problem is the follwing:

    .. math::

        \max_x p^T &x \\
        \text{s.t. } B&x \leq f \\
        &x \leq  d\\
        &x \geq 0

    Where :math:`p` are the ``profit``-entries, :math:`(B, f)` specify the packing
    constraints, and :math:`d` specifies the multiplicity constriants.
    All entries are assumed to be non-negative.
    """

    def __init__(self, profit, B, f, d=None):
        super().init(profit, A=None, b=None, B=B, f=f, d=d, kind="maximize")
