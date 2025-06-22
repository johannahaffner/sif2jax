import jax.numpy as jnp

from .eigen import EIGEN


class EIGENBLS(EIGEN):
    """EIGENBLS - Solving symmetric eigenvalue problems as systems of
    nonlinear equations.

    The problem is, given a symmetric matrix A, to find an orthogonal
    matrix Q and diagonal matrix D such that A = Q(T) D Q.

    Example B: a tridiagonal matrix with diagonals 2 and off diagonals -1

    Source: An idea by Nick Gould

    Least-squares version

    SIF input: Nick Gould, Nov 1992.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def _matrix(self):
        # Matrix B is tridiagonal with 2 on diagonal and -1 on off-diagonals
        diag = jnp.full(self.n, 2.0)
        off_diag = jnp.full(self.n - 1, -1.0)

        # Create the tridiagonal matrix
        matrix = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)

        return matrix
