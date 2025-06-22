import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class HILBERTA(AbstractUnconstrainedMinimisation):
    """Hilbert matrix problem.

    Unconstrained quadratic minimization problem using a Hilbert matrix.
    The Hilbert matrix is notorious for being badly conditioned, which makes
    this a challenging test problem for optimization algorithms.

    Source:
    K. Schittkowski,
    "More Test Examples for Nonlinear Programming Codes",
    Springer Verlag, Heidelberg, 1987.

    See also Buckley#19 (p. 59)

    SIF input: Ph. Toint, Dec 1989.

    Classification: QUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Other suggested values: 4, 5, 6, 10, works for any positive integer

    def objective(self, y, args):
        del args

        # From AMPL: sum {i in 1..N} x[i]*(sum {j in 1..N}A[i,j]*x[j])
        # where A[i,j] := 1/(i+j-1)

        # Create Hilbert matrix using the formula A[i,j] = 1/(i+j-1)
        i_indices = jnp.arange(1, self.n + 1)
        j_indices = jnp.arange(1, self.n + 1)
        i_grid, j_grid = jnp.meshgrid(i_indices, j_indices, indexing="ij")

        # Create the Hilbert matrix A[i,j] = 1/(i+j-1)
        hilbert_matrix = 1.0 / inexact_asarray(i_grid + j_grid - 1)

        # Compute the quadratic form: 0.5 * x^T * A * x
        # Note: Need 0.5 factor to match PyCUTEst results
        return 0.5 * jnp.dot(y, jnp.dot(hilbert_matrix, y))

    def y0(self):
        # From AMPL data section: x[1] = -4, x[2] = -2, but default in SIF is -3.0
        # Let's use the SIF default for consistency
        return inexact_asarray(jnp.full(self.n, -3.0))

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        return None

    def expected_objective_value(self):
        # The minimum value depends on the dimension
        return None
