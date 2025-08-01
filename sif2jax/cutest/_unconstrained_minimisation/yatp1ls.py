"""Yet another test problem involving double pseudo-stochastic constraints
on a square matrix. This is a least-squares formulation.

The problem involves finding a matrix X and vectors Y, Z such that:
- x_{ij}^3 - A x_{ij}^2 - (y_i + z_i)(x_{ij}cos(x_{ij}) - sin(x_{ij})) = 0
- sum_j sin(x_{ij})/x_{ij} = 1 for all i (row sums)
- sum_i sin(x_{ij})/x_{ij} = 1 for all j (column sums)

The problem is non convex.

Source: a late evening idea by Ph. Toint

SIF input: Ph. Toint, June 2003.

Classification: SUR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class YATP1LS(AbstractUnconstrainedMinimisation):
    """Yet Another Toint Problem 1 - Least Squares version."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    N: int = 350  # Matrix dimension (default from SIF)
    A: float = 10.0  # Constant in equations

    def __init__(self, N: int = 350, A: float = 10.0):
        self.N = N
        self.A = A

    @property
    def n(self):
        """Number of variables: N² + 2N."""
        return self.N * self.N + 2 * self.N

    @property
    def y0(self):
        """Initial guess."""
        # All X(i,j) = 1.0, Y(i) = 0.0, Z(i) = 0.0 (from START POINT section)
        y0 = jnp.zeros(self.n, dtype=jnp.float64)
        # Set X values (first N² elements) to 1.0
        y0 = y0.at[: self.N * self.N].set(1.0)
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
        1. x_{ij}^3 - A*x_{ij}^2 - (y_i + z_i)*(x_{ij}*cos(x_{ij}) - sin(x_{ij}))
           for all i,j
        2. sum_j sin(x_{ij})/x_{ij} - 1 for all i (row sums)
        3. sum_i sin(x_{ij})/x_{ij} - 1 for all j (column sums)
        """
        del args  # Not used

        x_end, y_start, y_end, z_start, z_end = self._get_indices()

        # Extract variables
        x_flat = y[:x_end]  # X matrix in flattened form
        y_vec = y[y_start:y_end]  # Y vector
        z_vec = y[z_start:z_end]  # Z vector

        # Reshape X to matrix form
        X = x_flat.reshape((self.N, self.N))

        # Compute residuals for E(i,j) groups
        e_residuals = []
        for i in range(self.N):
            for j in range(self.N):
                x_ij = X[i, j]
                y_i = y_vec[i]
                z_i = z_vec[i]

                # x_{ij}^3 - A*x_{ij}^2 - (y_i + z_i)*(x_{ij}*cos(x_{ij}) - sin(x_{ij}))
                term1 = x_ij**3
                term2 = -self.A * x_ij**2
                term3 = -(y_i + z_i) * (x_ij * jnp.cos(x_ij) - jnp.sin(x_ij))

                residual = term1 + term2 + term3
                e_residuals.append(residual)

        # Compute residuals for ER(i) groups (row sums)
        er_residuals = []
        for i in range(self.N):
            row_sum = jnp.array(0.0)
            for j in range(self.N):
                x_ij = X[i, j]
                # Avoid division by zero
                ratio = jnp.where(jnp.abs(x_ij) < 1e-15, 0.0, jnp.sin(x_ij) / x_ij)
                row_sum += ratio
            er_residuals.append(row_sum - 1.0)

        # Compute residuals for EC(j) groups (column sums)
        ec_residuals = []
        for j in range(self.N):
            col_sum = jnp.array(0.0)
            for i in range(self.N):
                x_ij = X[i, j]
                # Avoid division by zero
                ratio = jnp.where(jnp.abs(x_ij) < 1e-15, 0.0, jnp.sin(x_ij) / x_ij)
                col_sum += ratio
            ec_residuals.append(col_sum - 1.0)

        # Combine all residuals
        all_residuals = jnp.array(e_residuals + er_residuals + ec_residuals)

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
