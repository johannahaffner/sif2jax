import jax.numpy as jnp

from ..._misc import inexact_asarray
from .eigen import EIGEN


class EIGENALS(EIGEN):
    """EIGENALS - Solving symmetric eigenvalue problems as systems of
    nonlinear equations.

    The problem is, given a symmetric matrix A, to find an orthogonal
    matrix Q and diagonal matrix D such that A = Q(T) D Q.

    Example A: a diagonal matrix with eigenvalues 1, ..., N.

    Source: An idea by Nick Gould

    Least-squares version

    SIF input: Nick Gould, Nov 1992.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def _matrix(self):
        # Matrix A is diagonal with entries 1, 2, ..., n
        return inexact_asarray(jnp.diag(jnp.arange(1, self.n + 1)))
