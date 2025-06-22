import jax.numpy as jnp

from ..._misc import inexact_asarray
from .eigen import EIGEN


class EIGENCLS(EIGEN):
    """EIGENCLS - Solving symmetric eigenvalue problems as systems of
    nonlinear equations.

    The problem is, given a symmetric matrix A, to find an orthogonal
    matrix Q and diagonal matrix D such that A = Q(T) D Q.

    Example C: a tridiagonal matrix suggested by J. H. Wilkinson.

    Source: An idea by Nick Gould

    Least-squares version

    SIF input: Nick Gould, Nov 1992.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    m: int = 25  # Parameter M from SIF file
    n: int = 51  # Matrix dimension = 2*M+1 = 51

    def _matrix(self):
        # Matrix C is a Wilkinson tridiagonal matrix as defined in SIF file
        # A(1,1) = M = 25
        # A(J,J) = M+1-J for J = 2 to N
        # A(J-1,J) = 1.0 for super-diagonal elements

        # Create diagonal entries
        diag_values = jnp.zeros(self.n)
        diag_values = diag_values.at[0].set(self.m)  # A(1,1) = M = 25
        j_indices = jnp.arange(2, self.n + 1)  # J = 2 to N
        diag_values = diag_values.at[1:].set(
            inexact_asarray(self.m + 1) - inexact_asarray(j_indices)
        )  # A(J,J) = M+1-J

        # Create the off-diagonals (super and sub diagonals are 1.0)
        off_diag = jnp.ones(self.n - 1)

        # Create the tridiagonal matrix
        matrix = (
            jnp.diag(diag_values) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
        )

        return inexact_asarray(matrix)
