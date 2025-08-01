import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LUKSAN11(AbstractNonlinearEquations):
    """
    Problem 11 (chained serpentine) in the paper

    L. Luksan
    Hybrid methods in large sparse nonlinear least squares
    J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.

    classification NOR2-AN-V-V
    """

    s: int = 99  # Seed for dimensions
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, s: int = 99):
        self.s = s

    @property
    def n(self) -> int:
        """Number of variables: S + 1."""
        return self.s + 1

    def num_residuals(self) -> int:
        """Number of residuals: 2 * S."""
        return 2 * self.s

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.full(self.n, -0.8, dtype=jnp.float64)

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector."""
        x = y
        s = self.s
        residuals = []

        # Build residuals in pairs
        for i in range(s):
            # First residual of pair: 20*x[i]/(1+x[i]^2) - 10*x[i+1]
            xi = x[i]
            d = 1.0 + xi * xi
            res1 = 20.0 * xi / d - 10.0 * x[i + 1]
            residuals.append(res1)

            # Second residual of pair: x[i] (constant RHS = 0 for even indices)
            # Note: Only odd indices have RHS = 1.0
            res2 = x[i]
            residuals.append(res2)

        residuals = jnp.array(residuals, dtype=jnp.float64)

        # Apply constants: E(i+1) = 1.0 for i that are multiples of 2
        # In 0-indexed, this means residuals at indices 1, 3, 5, ... have RHS = 1.0
        for i in range(1, 2 * s, 2):
            residuals = residuals.at[i].add(-1.0)

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
        # Solution should satisfy residuals = 0
        return jnp.zeros(self.n, dtype=jnp.float64)

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
        """Free bounds for all variables."""
        return None
