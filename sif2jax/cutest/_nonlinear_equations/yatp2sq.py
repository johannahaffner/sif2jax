"""Another test problem involving double pseudo-stochastic constraints
on a square matrix. This is a nonlinear equations formulation.

The problem involves finding a matrix X and vectors Y, Z such that:
- x_{ij} - (y_i + z_i)(1 + cos(x_{ij})) = A for all i,j
- sum_i (x_{ij} + sin(x_{ij})) = 1 for all j (column sums)
- sum_j (x_{ij} + sin(x_{ij})) = 1 for all i (row sums)

The problem is non convex.

Source: a late evening idea by Ph. Toint

SIF input: Ph. Toint, June 2003.

Classification: NOR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class YATP2SQ(AbstractNonlinearEquations):
    """YATP2SQ - nonlinear equations version."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    _N: int = 350  # Matrix dimension (default from SIF)
    A: float = 1.0  # Constant in equations (default from SIF)

    def __init__(self, N: int = 350, A: float = 1.0):
        self._N = N
        self.A = A

    @property
    def n(self):
        """Number of variables: N^2 + 2N."""
        return self._N * self._N + 2 * self._N

    @property
    def m(self):
        """Number of equations: N^2 + 2N."""
        return self._N * self._N + 2 * self._N

    @property
    def y0(self):
        """Initial guess."""
        # All X(i,j) = 1.0, Y(i) = 0.0, Z(i) = 0.0 (from START POINT section)
        y0 = jnp.zeros(self.n, dtype=jnp.float64)
        # Set X values (first N^2 elements) to 1.0
        y0 = y0.at[: self._N * self._N].set(1.0)
        return y0

    @property
    def args(self):
        """No additional arguments."""
        return None

    def _get_indices(self):
        """Get indices for X matrix, Y and Z vectors."""
        n_sq = self._N * self._N
        x_end = n_sq
        y_start = n_sq
        y_end = n_sq + self._N
        z_start = y_end
        z_end = n_sq + 2 * self._N
        return x_end, y_start, y_end, z_start, z_end

    def constraint(self, y):
        """Compute the nonlinear equations.

        The equations are:
        1. x_{ij} - (y_i + z_i)*(1 + cos(x_{ij})) - A = 0 for all i,j
        2. sum_i (x_{ij} + sin(x_{ij})) - 1 = 0 for all j (column sums)
        3. sum_j (x_{ij} + sin(x_{ij})) - 1 = 0 for all i (row sums)
        """
        x_end, y_start, y_end, z_start, z_end = self._get_indices()

        # Extract variables
        x_flat = y[:x_end]  # X matrix in flattened form
        y_vec = y[y_start:y_end]  # Y vector
        z_vec = y[z_start:z_end]  # Z vector

        # Reshape X to matrix form
        X = x_flat.reshape((self._N, self._N))

        equations = []

        # E(i,j) equations: x_{ij} - (y_i + z_i)*(1 + cos(x_{ij})) - A = 0
        for i in range(self._N):
            for j in range(self._N):
                x_ij = X[i, j]
                y_i = y_vec[i]
                z_i = z_vec[i]

                equation = x_ij - (y_i + z_i) * (1.0 + jnp.cos(x_ij)) - self.A
                equations.append(equation)

        # EC(j) equations: sum_i (x_{ij} + sin(x_{ij})) - 1 = 0 for all j (column sums)
        for j in range(self._N):
            col_sum = jnp.array(0.0)
            for i in range(self._N):
                x_ij = X[i, j]
                col_sum += x_ij + jnp.sin(x_ij)
            equations.append(col_sum - 1.0)

        # ER(i) equations: sum_j (x_{ij} + sin(x_{ij})) - 1 = 0 for all i (row sums)
        for i in range(self._N):
            row_sum = jnp.array(0.0)
            for j in range(self._N):
                x_ij = X[i, j]
                row_sum += x_ij + jnp.sin(x_ij)
            equations.append(row_sum - 1.0)

        # Convert to JAX array
        eq_constraints = jnp.array(equations)
        ineq_constraints = None

        return eq_constraints, ineq_constraints

    @property
    def bounds(self):
        """No explicit bounds."""
        return None

    @property
    def expected_result(self):
        """Expected result - not provided in SIF."""
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
    def expected_objective_value(self):
        """Expected objective value at the solution."""
        return jnp.array(0.0)