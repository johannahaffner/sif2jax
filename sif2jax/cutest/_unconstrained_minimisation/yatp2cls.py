"""Another test problem involving double pseudo-stochastic constraints
on a square matrix. This is a corrected least-squares formulation.

The problem involves finding a matrix X and vectors Y, Z such that:
- x_{ij} - (y_i + z_j)(1 + cos(x_{ij})) = A for all i,j (corrected: z_j)
- sum_i (x_{ij} + sin(x_{ij})) = 1 for all j (column sums)
- sum_j (x_{ij} + sin(x_{ij})) = 1 for all i (row sums)

Key correction from YATP2LS: z_j instead of z_i in the first equation.

The problem is non convex.

Source: a late evening idea by Ph. Toint

SIF input: Ph. Toint, June 2003.
           corrected Nick Gould, March 2019

Classification: SUR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class YATP2CLS(AbstractUnconstrainedMinimisation):
    """Yet Another Toint Problem 2 - Corrected Least Squares version."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    N: int = 350  # Matrix dimension (default from SIF)
    A: float = 1.0  # Constant in equations (default from SIF)

    def __init__(self, N: int = 350, A: float = 1.0):
        self.N = N
        self.A = A

    @property
    def n(self):
        """Number of variables: N^2 + 2N."""
        return self.N * self.N + 2 * self.N

    @property
    def y0(self):
        """Initial guess."""
        # All X(i,j) = 10.0, Y(i) = 0.0, Z(i) = 0.0 (from START POINT section)
        y0 = jnp.zeros(self.n, dtype=jnp.float64)
        # Set X values (first N^2 elements) to 10.0
        y0 = y0.at[: self.N * self.N].set(10.0)
        return y0

    @property
    def args(self):
        """No additional arguments."""
        return None

    def _get_indices(self):
        """Get indices for X matrix, Y and Z vectors."""
        n_sq = self.N * self.N
        x_end = n_sq
        y_start = n_sq
        y_end = n_sq + self.N
        z_start = y_end
        z_end = n_sq + 2 * self.N
        return x_end, y_start, y_end, z_start, z_end

    def objective(self, y, args):
        """Compute the least squares objective function.

        The objective is the sum of squares of:
        1. x_{ij} - (y_i + z_j)*(1 + cos(x_{ij})) - A for all i,j (corrected)
        2. sum_i (x_{ij} + sin(x_{ij})) - 1 for all j (column sums)
        3. sum_j (x_{ij} + sin(x_{ij})) - 1 for all i (row sums)
        """
        del args  # Not used

        x_end, y_start, y_end, z_start, z_end = self._get_indices()

        # Extract variables
        x_flat = y[:x_end]  # X matrix in flattened form
        y_vec = y[y_start:y_end]  # Y vector
        z_vec = y[z_start:z_end]  # Z vector

        # Reshape X to matrix form
        X = x_flat.reshape((self.N, self.N))

        # Vectorized computation of E(i,j) residuals
        # Broadcast y_i to match X dimensions (rows)
        y_i_broadcast = y_vec[:, jnp.newaxis]  # Shape: (N, 1)
        # Broadcast z_j to match X dimensions (columns)
        z_j_broadcast = z_vec[jnp.newaxis, :]  # Shape: (1, N)

        # Compute all E(i,j) residuals at once
        e_residuals = (
            X - (y_i_broadcast + z_j_broadcast) * (1.0 + jnp.cos(X)) - self.A
        ).flatten()

        # Vectorized computation for column and row sums
        X_plus_sinX = X + jnp.sin(X)

        # Compute EC(j) residuals (column sums)
        ec_residuals = jnp.sum(X_plus_sinX, axis=0) - 1.0

        # Compute ER(i) residuals (row sums)
        er_residuals = jnp.sum(X_plus_sinX, axis=1) - 1.0

        # Combine all residuals
        all_residuals = jnp.concatenate([e_residuals, ec_residuals, er_residuals])

        # Return sum of squares
        return jnp.sum(all_residuals**2)

    @property
    def expected_result(self):
        """Expected result - not provided in SIF."""
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
    def expected_objective_value(self):
        """Expected objective value at the solution."""
        return jnp.array(0.0)
