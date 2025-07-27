"""A non-convex problem involving double pseudo-stochastic constraints
on a square matrix. This is a least-squares formulation.

The problem involves finding a matrix X and vectors Y, Z such that:
- x_{ij} - y_i - z_j - (y_i + z_i) * cos(x_{ij}) = A for all i,j
- sum_j (x_{ij} + sin(x_{ij})) = 1 for all i (row sums)
- sum_j (x_{ij} + sin(x_{ij})) = 1 for all i (duplicate row sums)

Note: The SIF file has a complex structure where the linear part includes
Y(I) and Z(J), but the nonlinear ATP2 element uses Y(I) and Z(I).
Additionally, the SIF file appears to have a bug where EC(I) is used
instead of EC(J) in the group definitions, making both ER and EC compute
row sums instead of having one compute column sums.

Source: Ph. Toint, June 2003.

SIF input: Ph. Toint, June 2003.

Classification: SUR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class YATP2LS(AbstractUnconstrainedMinimisation):
    """Yet Another Toint Problem 2 - Least Squares version."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    N: int = 2  # Matrix dimension (default from SIF, can be 2, 10, 50, 100, 350)
    A: float = 1.0  # Constant in equations

    @property
    def n(self):
        """Number of variables: N² + 2N."""
        return self.N * self.N + 2 * self.N

    @property
    def y0(self):
        """Initial guess."""
        # All X(i,j) = 10.0, Y(i) = 0.0, Z(i) = 0.0
        y0 = jnp.zeros(self.n)
        # Set X values (first N² elements) to 10.0
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
        1. x_{ij} - y_i - z_j - (y_i + z_i) * cos(x_{ij}) - A for all i,j
        2. sum_j (x_{ij} + sin(x_{ij})) - 1 for all i (row sums, ER groups)
        3. sum_j (x_{ij} + sin(x_{ij})) - 1 for all i (duplicate row sums, EC groups)
        """
        del args  # Not used

        x_end, y_start, y_end, z_start, z_end = self._get_indices()

        # Extract variables
        x_flat = y[:x_end]  # X matrix in flattened form
        y_vec = y[y_start:y_end]  # Y vector
        z = y[z_start:z_end]  # Z vector

        # Reshape X to matrix form
        x = x_flat.reshape((self.N, self.N))

        equations = []

        # Type 1 equations: x_{ij} - y_i - z_j - (y_i + z_i) * cos(x_{ij}) = A
        # Note: The SIF file has linear terms for Y(I) and Z(J) with coeff -1,
        # but the ATP2 element DC(I,J) uses Y(I) and Z(I)
        for i in range(self.N):
            for j in range(self.N):
                x_ij = x[i, j]
                # Linear part: x_{ij} - y_i - z_j
                # Nonlinear part: -(y_i + z_i) * cos(x_{ij})
                eq = x_ij - y_vec[i] - z[j] - (y_vec[i] + z[i]) * jnp.cos(x_ij) - self.A
                equations.append(eq)

        # Type 2 equations: sum_j (x_{ij} + sin(x_{ij})) = 1 for all i (row sums)
        # Note: The SIF file has a bug where EC(I) is used instead of EC(J) in line 70,
        # so ER and EC both sum over j for each i (both are row sums)
        for i in range(self.N):
            row_sum = 0.0
            for j in range(self.N):
                x_ij = x[i, j]
                row_sum += x_ij + jnp.sin(x_ij)
            eq = row_sum - 1.0
            equations.append(eq)

        # Type 3 equations: Also sum_j (x_{ij} + sin(x_{ij})) = 1 for all i
        # Due to the SIF file bug, EC groups are identical to ER groups
        for i in range(self.N):
            row_sum = 0.0
            for j in range(self.N):
                x_ij = x[i, j]
                row_sum += x_ij + jnp.sin(x_ij)
            eq = row_sum - 1.0
            equations.append(eq)

        # Sum of squares
        equations_array = jnp.array(equations)
        return jnp.sum(equations_array**2)

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None
