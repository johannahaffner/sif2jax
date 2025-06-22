import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class HILBERTB(AbstractUnconstrainedMinimisation):
    """Perturbed Hilbert matrix problem.

    Unconstrained quadratic minimization problem using a Hilbert matrix
    with a diagonal perturbation to improve conditioning. The Hilbert matrix
    is notorious for being badly conditioned, and this perturbation makes
    the problem more tractable.

    Source: problem 19 (p. 59) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: QUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 10  # Other suggested values in SIF: 5, 50, should work for n >= 1

    def objective(self, y, args):
        del args

        # From AMPL: sum {i in 1..N} (sum {j in 1..i-1} x[i]*x[j]/(i+j-1)
        # + (x[i]^2)*(D+1/(4*i-2)))

        d = 5.0  # Parameter D from AMPL

        # Vectorized computation using JAX operations
        # Create indices for the upper triangular part (j < i)
        i_indices = jnp.arange(1, self.n + 1)
        j_indices = jnp.arange(1, self.n + 1)
        i_grid, j_grid = jnp.meshgrid(i_indices, j_indices, indexing="ij")

        # Mask for upper triangular part (j < i)
        upper_triangular_mask = j_grid < i_grid

        # Compute off-diagonal terms: x[i]*x[j]/(i+j-1) for j < i
        y_i = y[i_grid - 1]  # Convert to 0-based indexing
        y_j = y[j_grid - 1]  # Convert to 0-based indexing
        hilbert_coeffs = 1.0 / inexact_asarray(i_grid + j_grid - 1)

        off_diagonal_terms = jnp.where(
            upper_triangular_mask, y_i * y_j * hilbert_coeffs, 0.0
        )

        # Diagonal terms: (x[i]^2)*(D+1/(4*i-2))
        diagonal_coeffs = d + 1.0 / inexact_asarray(4 * i_indices - 2)
        diagonal_terms = y**2 * diagonal_coeffs

        return jnp.sum(jnp.asarray(off_diagonal_terms)) + jnp.sum(diagonal_terms)

    def y0(self):
        # Starting point: all variables set to -3.0 (from AMPL var x{1..N} := -3.0)
        return jnp.full(self.n, -3.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is x = 0 for a positive definite quadratic form
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # The minimum value of the quadratic form is 0.0
        return jnp.array(0.0)
