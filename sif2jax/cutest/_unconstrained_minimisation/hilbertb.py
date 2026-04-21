import jax.numpy as jnp

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

        # Vectorized computation using broadcasting
        n = self.n
        i_indices = jnp.arange(1, n + 1, dtype=y.dtype)

        # Hilbert coefficients: 1/(i+j-1) for i,j in 1..n
        i_grid = i_indices[:, None]  # (n, 1)
        j_grid = i_indices[None, :]  # (1, n)
        hilbert_coeffs = 1.0 / (i_grid + j_grid - 1)  # (n, n)

        # Off-diagonal: y[i]*y[j]/(i+j-1) for j < i (upper triangular)
        y_outer = y[:, None] * y[None, :]  # (n, n)
        upper_mask = j_grid < i_grid
        off_diagonal = jnp.sum(jnp.where(upper_mask, y_outer * hilbert_coeffs, 0.0))  # pyright: ignore[reportArgumentType]

        # Diagonal: y[i]^2 * (D + 1/(4i-2))
        diagonal_coeffs = d + 1.0 / (4 * i_indices - 2)
        diagonal = jnp.sum(y**2 * diagonal_coeffs)

        return off_diagonal + diagonal

    @property
    def y0(self):
        # Starting point: all variables set to -3.0 (from AMPL var x{1..N} := -3.0)
        return jnp.full(self.n, -3.0)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The optimal solution is x = 0 for a positive definite quadratic form
        return jnp.zeros(self.n)

    @property
    def expected_objective_value(self):
        # The minimum value of the quadratic form is 0.0
        return jnp.array(0.0)
