import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class INTEGREQ(AbstractNonlinearEquations):
    """
    The discrete integral problem

    Source:  Problem 29 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Feb 1990.

    classification NOR2-AN-V-V
    """

    n: int = 500  # Number of free variables (default from SIF)
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return self.n  # G(1) to G(N)

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        # N+2 discretization points: x(0) to x(N+1)
        # x(0) and x(N+1) are fixed at 0
        h = 1.0 / (self.n + 1)

        # Vectorized starting values for x(1) to x(N)
        i = jnp.arange(1, self.n + 1, dtype=jnp.float64)
        t_i = i * h
        x_middle = t_i * (t_i - 1.0)

        # Construct full x array with x(0) = 0, x(1..N) = computed values, x(N+1) = 0
        x = jnp.concatenate([jnp.array([0.0]), x_middle, jnp.array([0.0])])

        return x

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        The problem discretizes an integral equation. The residuals are:
        G(i): x(i) + sum of weighted cubic terms = 0, for i = 1..N

        Note: x(0) and x(N+1) are fixed at 0 and not part of the equations.
        """
        x = y
        n = self.n
        h = 1.0 / (n + 1)
        halfh = 0.5 * h

        # Create index arrays
        i_indices = jnp.arange(1, n + 1, dtype=jnp.float64)
        j_indices = jnp.arange(1, n + 1, dtype=jnp.float64)

        # Compute t_i and t_j for all indices
        t_i = i_indices * h
        t_j = j_indices * h

        # Compute the cubic terms for all j: (x(j) + (1 + t_j))^3
        vplusb = x[1 : n + 1] + (1.0 + t_j)
        cubic_terms = vplusb**3

        # Create masks for lower and upper triangular parts
        # lower_mask[i-1, j-1] = True if j <= i (for the first sum)
        # upper_mask[i-1, j-1] = True if j > i (for the second sum)
        i_grid, j_grid = jnp.meshgrid(i_indices, j_indices, indexing="ij")
        lower_mask = j_grid <= i_grid
        upper_mask = j_grid > i_grid

        # Compute weights for lower triangular part (first sum)
        # w_il = (1.0 - t_i) * halfh * t_j for j <= i
        p1 = (1.0 - t_i[:, None]) * halfh  # Shape: (n, 1)
        w_il = p1 * t_j[None, :]  # Shape: (n, n)
        w_il = jnp.where(lower_mask, w_il, 0.0)

        # Compute weights for upper triangular part (second sum)
        # w_iu = t_i * halfh * (1.0 - t_j) for j > i
        p2 = t_i[:, None] * halfh  # Shape: (n, 1)
        w_iu = p2 * (1.0 - t_j[None, :])  # Shape: (n, n)
        w_iu = jnp.where(upper_mask, w_iu, 0.0)

        # Combine weights
        weights = w_il + w_iu  # Shape: (n, n)

        # Compute weighted sums
        weighted_cubic_sums = jnp.dot(weights, cubic_terms)

        # Add x(i) to get final residuals
        residuals = x[1 : n + 1] + weighted_cubic_sums

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return ()

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """Bounds for all variables. x(0) and x(N+1) are fixed at 0."""
        # Create bounds arrays
        lower = jnp.full(self.n + 2, -jnp.inf, dtype=jnp.float64)
        upper = jnp.full(self.n + 2, jnp.inf, dtype=jnp.float64)

        # x(0) and x(N+1) are fixed at 0
        lower = lower.at[0].set(0.0)
        upper = upper.at[0].set(0.0)
        lower = lower.at[self.n + 1].set(0.0)
        upper = upper.at[self.n + 1].set(0.0)

        return lower, upper

    @property
    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Solution: x(0) = 0, x(N+1) = 0, other x values satisfy equations
        return jnp.zeros(self.n + 2, dtype=jnp.float64)

    @property
    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
