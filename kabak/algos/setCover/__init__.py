"""
Algorithms for Set Cover problems.

Introduction
============

Set cover problems are special cases of covering problems. They sometimes support
improved algorithmic performance guarantees, especially when the sets follow geometric structure.
"""

from kabak.algos.setCover.roundingGeometric import geometricGreedy

__all__ = ["geometricGreedy"]
