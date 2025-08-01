"""Yet another test problem involving double pseudo-stochastic constraints
on a square matrix. This is a corrected nonlinear equations formulation.

The problem involves finding a matrix X and vectors Y, Z such that:
- x_{ij}^3 - A x_{ij}^2 - (y_i + z_j)(x_{ij}cos(x_{ij}) - sin(x_{ij})) = 0
- sum_j sin(x_{ij})/x_{ij} = 1 for all i (row sums)
- sum_i sin(x_{ij})/x_{ij} = 1 for all j (column sums)

Key correction from YATP1NE: z_j instead of z_i in the first equation.

The problem is non convex.

Source: a late evening idea by Ph. Toint

SIF input: Ph. Toint, June 2003.
           corrected Nick Gould, March 2019

Classification: NOR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class YATP1CNE(AbstractNonlinearEquations):
    """YATP1CNE - Corrected nonlinear equations version."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    _N: int = 350  # Matrix dimension (default from SIF)
    A: float = 10.0  # Constant in equations

    def __init__(self, N: int = 350, A: float = 10.0):
        self._N = N
        self.A = A

    @property
    def n(self):
        """Number of variables: N² + 2N."""
        return self._N * self._N + 2 * self._N

    @property
    def m(self):
        """Number of equations: N² + 2N."""
        return self._N * self._N + 2 * self._N

    @property
    def y0(self):
        """Initial guess."""
        # All X(i,j) = 1.0, Y(i) = 0.0, Z(i) = 0.0 (from START POINT section)
        y0 = jnp.zeros(self.n, dtype=jnp.float64)
        # Set X values (first N² elements) to 1.0
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
        1. x_{ij}^3 - A*x_{ij}^2 - (y_i + z_j)*(x_{ij}*cos(x_{ij}) - sin(x_{ij})) = 0
           for all i,j (corrected: z_j instead of z_i)
        2. sum_j sin(x_{ij})/x_{ij} - 1 = 0 for all i (row sums)
        3. sum_i sin(x_{ij})/x_{ij} - 1 = 0 for all j (column sums)
        """
        x_end, y_start, y_end, z_start, z_end = self._get_indices()

        # Extract variables
        x_flat = y[:x_end]  # X matrix in flattened form
        y_vec = y[y_start:y_end]  # Y vector
        z_vec = y[z_start:z_end]  # Z vector

        # Reshape X to matrix form
        X = x_flat.reshape((self._N, self._N))

        equations = []

        # E(i,j) equations: x_{ij}^3 - A*x_{ij}^2 - (y_i + z_j)*(x_{ij}*cos(x_{ij})
        # - sin(x_{ij})) = 0
        for i in range(self._N):
            for j in range(self._N):
                x_ij = X[i, j]
                y_i = y_vec[i]
                z_j = z_vec[j]  # Key correction: z_j instead of z_i

                term1 = x_ij**3
                term2 = -self.A * x_ij**2
                term3 = -(y_i + z_j) * (x_ij * jnp.cos(x_ij) - jnp.sin(x_ij))

                equation = term1 + term2 + term3
                equations.append(equation)

        # ER(i) equations: sum_j sin(x_{ij})/x_{ij} - 1 = 0 for all i (row sums)
        for i in range(self._N):
            row_sum = jnp.array(0.0)
            for j in range(self._N):
                x_ij = X[i, j]
                # Avoid division by zero
                ratio = jnp.where(jnp.abs(x_ij) < 1e-15, 0.0, jnp.sin(x_ij) / x_ij)
                row_sum += ratio
            equations.append(row_sum - 1.0)

        # EC(j) equations: sum_i sin(x_{ij})/x_{ij} - 1 = 0 for all j (column sums)
        for j in range(self._N):
            col_sum = jnp.array(0.0)
            for i in range(self._N):
                x_ij = X[i, j]
                # Avoid division by zero
                ratio = jnp.where(jnp.abs(x_ij) < 1e-15, 0.0, jnp.sin(x_ij) / x_ij)
                col_sum += ratio
            equations.append(col_sum - 1.0)

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