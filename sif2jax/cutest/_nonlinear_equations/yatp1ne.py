"""Yet another test problem involving double pseudo-stochastic constraints
on a square matrix. This is a nonlinear equations formulation.

The problem involves finding a matrix X and vectors Y, Z such that:
- x_{ij}^3 - A x_{ij}^2 - (y_i + z_i)(x_{ij}cos(x_{ij}) - sin(x_{ij})) = 0
- sum_j sin(x_{ij})/x_{ij} = 1 for all i (row sums)
- sum_i sin(x_{ij})/x_{ij} = 1 for all j (column sums)

The problem is non convex.

Source: a late evening idea by Ph. Toint

SIF input: Ph. Toint, June 2003.

Classification: NOR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class YATP1NE(AbstractNonlinearEquations):
    """Yet Another Toint Problem 1 - Nonlinear Equations version."""

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
        """Number of constraints: N² + 2N."""
        return self._N * self._N + 2 * self._N

    @property
    def y0(self):
        """Initial guess."""
        # All X(i,j) = 6.0, Y(i) = 0.0, Z(i) = 0.0 (from START POINT section)
        y0 = jnp.zeros(self.n, dtype=jnp.float64)
        # Set X values (first N² elements) to 6.0
        y0 = y0.at[: self._N * self._N].set(6.0)
        return y0

    @property
    def bounds(self):
        """No explicit bounds."""
        return None

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
        1. x_{ij}^3 - A*x_{ij}^2 - (y_i + z_i)*(x_{ij}*cos(x_{ij}) - sin(x_{ij})) = 0
           for all i,j
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

        # E(i,j) equations: vectorized version
        # x_{ij}^3 - A*x_{ij}^2 - (y_i + z_i)*(x_{ij}*cos(x_{ij}) - sin(x_{ij})) = 0
        term1 = X**3
        term2 = -self.A * X**2

        # Broadcast y_i + z_i across columns for each row
        y_plus_z = (y_vec + z_vec)[:, jnp.newaxis]  # Shape: (N, 1)
        term3 = -y_plus_z * (X * jnp.cos(X) - jnp.sin(X))

        E_equations = (term1 + term2 + term3).ravel()

        # Compute sin(x)/x with proper handling of x=0 (sinc function)
        sinc_X = jnp.where(jnp.abs(X) < 1e-15, 1.0, jnp.sin(X) / X)

        # ER(i) equations: sum_j sin(x_{ij})/x_{ij} - 1 = 0 for all i (row sums)
        ER_equations = jnp.sum(sinc_X, axis=1) - 1.0  # type: ignore

        # EC(j) equations: sum_i sin(x_{ij})/x_{ij} - 1 = 0 for all j (column sums)
        EC_equations = jnp.sum(sinc_X, axis=0) - 1.0  # type: ignore

        # Concatenate all equations
        eq_constraints = jnp.concatenate([E_equations, ER_equations, EC_equations])
        ineq_constraints = None

        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        """Expected result - not provided in SIF."""
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
    def expected_objective_value(self):
        """Expected objective value at the solution."""
        return jnp.array(0.0)
