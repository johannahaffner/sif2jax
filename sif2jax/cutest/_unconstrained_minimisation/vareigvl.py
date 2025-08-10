import jax
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed
# Attempts made: Fixed ConcretizationTypeError by using masks and dynamic_slice
# Suspected issues: Matrix computation discrepancy - obj/grad/Hessian all fail
# Resources needed: Check original SIF file, verify banded matrix computation
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
    M: int = 4  # Half bandwidth (2M+1 = 9 from description)
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

        # Create index arrays
        i_indices = jnp.arange(self.N)

        # Create a padded version of x for easier indexing
        # Pad with zeros on both sides for the band computation
        x_padded = jnp.pad(x, (self.M, self.M), mode="constant", constant_values=0.0)

        # For each row i, compute (Ax)_i - mu * x_i using a sliding window
        def compute_row_residual(i):
            # Create relative indices for the band (from -M to M)
            relative_j = jnp.arange(-self.M, self.M + 1)

            # Get actual j indices (clipped to valid range)
            j_indices = i + relative_j

            # Create mask for valid indices
            mask = (j_indices >= 0) & (j_indices < self.N)

            # Compute matrix elements for the band
            # Use (i+1) for 1-based indexing in formula
            ij = (i + 1) * (j_indices + 1)
            sij = jnp.sin(ij)
            j_minus_i = relative_j
            arg = -(j_minus_i**2) / n2
            exp_arg = jnp.exp(arg)
            aij = jnp.where(mask, sij * exp_arg, 0.0)

            # Get corresponding x values from padded array using dynamic_slice
            # In padded array, index i corresponds to position i
            # (since we padded with M on left)
            x_band = jax.lax.dynamic_slice(x_padded, [i], [2 * self.M + 1])

            # Compute row sum
            row_sum = jnp.sum(aij * x_band)

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
