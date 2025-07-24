from typing_extensions import override

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# Note: In SIF least-squares problems, pycutest interprets groups without
# explicit type assignments as linear terms, not squared terms. This differs
# from the typical least-squares interpretation where residuals are squared.
class TENFOLDTRLS(AbstractUnconstrainedMinimisation):
    """The 10FOLDTRLS function.

    The ten-fold triangular system whose root at zero has multiplicity 10.

    Source: Problem 8.3 in
    Wenrui Hao, Andrew J. Sommese and Zhonggang Zeng,
    "An algorithm and software for computing multiplicity structures
     at zeros of nonlinear systems", Technical Report,
    Department of Applied & Computational Mathematics & Statistics,
    University of Notre Dame, Indiana, USA (2012)

    SIF input: Nick Gould, Jan 2012.
    Least-squares version of 10FOLDTR.SIF, Nick Gould, Jun 2024.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 1000  # Problem dimension, SIF file suggests 4, 10, 100, or 1000

    @property
    @override
    def name(self):
        return "10FOLDTRLS"

    def objective(self, y, args):
        del args

        # For each i, compute the sum of elements from x_1 to x_i
        # This implements the triangular system described in the SIF file
        # Where E(i) = sum_{j=1}^i x_j for i=1...n
        e = jnp.cumsum(y)

        # This is a least-squares problem (classification starts with 'S')
        # In pycutest's interpretation, groups without explicit types contribute
        # linearly, not as squared terms. Only groups with explicit types use
        # their specified functions.
        # E(1) through E(N-2) contribute linearly (no explicit type)
        # E(N-1) contributes as fourth power (L4 group type)
        # E(N) contributes as tenth power (L10 group type)

        # Sum of E(1) through E(N-2) - linear contribution
        f_linear = jnp.sum(e[: self.n - 2])

        # Special terms
        f1 = e[self.n - 2] ** 4  # E_{n-1}^4
        f2 = e[self.n - 1] ** 10  # E_n^10

        return f_linear + f1 + f2

    @property
    def y0(self):
        # Initial value of 10.0 as specified in the SIF file
        return jnp.full(self.n, 10.0)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The solution is at zero as mentioned in the SIF file description
        return jnp.zeros(self.n)

    @property
    def expected_objective_value(self):
        return jnp.array(0.0)
