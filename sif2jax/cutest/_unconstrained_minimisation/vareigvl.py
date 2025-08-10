import jax
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class VAREIGVL(AbstractUnconstrainedMinimisation):
    """Variational eigenvalue problem by Auchmuty.

    This problem features a banded matrix of bandwidth 2M+1 = 9.
    It has N least-squares groups, each having a linear part only
    and N nonlinear elements, plus a least q-th power group having
    N nonlinear elements.

    Source: problem 1 in
    J.J. More',
    "A collection of nonlinear model problems"
    Proceedings of the AMS-SIAM Summer seminar on the Computational
    Solution of Nonlinear Systems of Equations, Colorado, 1988.
    Argonne National Laboratory MCS-P60-0289, 1989.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January, 1995
    and Nick Gould, December, 2019, May 2024

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimension
    N: int = 4999  # Default from SIF file
    M: int = 6  # Half bandwidth (must be at most N)
    Q: float = 1.5  # Power parameter (must be in (1,2])

    @property
    def n(self):
        """Number of variables (N + 1 including mu)."""
        return self.N + 1

    def _compute_matrix_element(self, i, j):
        """Compute the matrix element A[i,j]."""
        n2 = self.N * self.N
        ij = (i + 1) * (j + 1)  # Convert to 1-based indexing
        sij = jnp.sin(ij)
        j_minus_i = (j + 1) - (i + 1)
        arg = -(j_minus_i**2) / n2
        exp_arg = jnp.exp(arg)
        return sij * exp_arg

    def objective(self, y, args=None):
        """Compute the objective function.

        y has N+1 elements: x[0..N-1] and mu
        """
        del args

        # Extract x and mu
        x = y[: self.N]
        mu = y[self.N]

        # Vectorized computation of Ax - mu*x
        n2 = self.N * self.N
        residuals = []

        # Create index arrays for vectorization
        i_indices = jnp.arange(self.N)

        # For each row i, compute (Ax)_i - mu * x_i
        def compute_row_residual(i):
            # Determine the band range for row i
            j_start = jnp.maximum(0, i - self.M)
            j_end = jnp.minimum(self.N, i + self.M + 1)

            # Compute matrix elements for the band
            j_indices = jnp.arange(j_start, j_end)
            ij = (i + 1) * (j_indices + 1)
            sij = jnp.sin(ij)
            j_minus_i = j_indices - i
            arg = -(j_minus_i**2) / n2
            exp_arg = jnp.exp(arg)
            aij = sij * exp_arg

            # Compute row sum
            row_sum = jnp.sum(aij * x[j_start:j_end])

            # Return residual
            return row_sum - mu * x[i]

        # Vectorize over all rows
        residuals = jax.vmap(compute_row_residual)(i_indices)

        # Compute sum of squared residuals
        objective_value = jnp.sum(residuals**2)

        # Add the least q-th power group
        x_norm_squared = jnp.sum(x**2)
        objective_value += x_norm_squared ** (self.Q / 2)

        return objective_value

    @property
    def y0(self):
        """Initial guess: x_i = 1.0, mu = 0.0."""
        x0 = jnp.ones(self.N)
        mu0 = jnp.array(0.0)
        return jnp.concatenate([x0, jnp.array([mu0])])

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """No bounds for this problem."""
        return None

    @property
    def expected_result(self):
        """Expected solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value is 0."""
        return jnp.array(0.0)
