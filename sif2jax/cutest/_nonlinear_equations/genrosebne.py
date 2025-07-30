import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class GENROSEBNE(AbstractNonlinearEquations):
    """
    The generalized Rosenbrock function.

    Source: problem 5 in
    S. Nash,
    "Newton-type minimization via the Lanczos process",
    SIAM J. Num. Anal. 21, 1984, 770-788.

    SIF input: Nick Gould, Oct 1992.
              minor correction by Ph. Shott, Jan 1995.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    version with simple bound constraints

    classification NOR2-AN-V-V
    """

    n: int = 500  # Default value from SIF
    m: int = 999  # 2*(n-1) equations
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        # From SIF: X(I) = I/(N+1)
        i_vals = jnp.arange(1, self.n + 1, dtype=jnp.float64)
        return i_vals / (self.n + 1)

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the generalized Rosenbrock problem"""
        x = y
        n = self.n

        # Initialize residuals - we have 2*(n-1) equations
        residuals = jnp.zeros(2 * (n - 1), dtype=jnp.float64)

        # For each i from 2 to n, we have 2 equations
        for i in range(1, n):  # i = 2 to n in 1-based indexing
            # Q(i) equation with SCALE 0.1
            # Element MSQR: -V^2 where V = x(i-1)
            # SCALE parameter divides the residual
            residuals = residuals.at[2 * (i - 1)].set((x[i] - x[i - 1] ** 2) / 0.1)

            # L(i) equation: x(i) - 1.0 = 0
            residuals = residuals.at[2 * (i - 1) + 1].set(x[i] - 1.0)

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Solution would be all ones
        return jnp.ones(self.n, dtype=jnp.float64)

    @property
    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """Bounds from the SIF file."""
        lower = jnp.full(self.n, 0.2, dtype=jnp.float64)
        upper = jnp.full(self.n, 0.5, dtype=jnp.float64)
        return lower, upper
