import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LUKSAN12(AbstractNonlinearEquations):
    """
    Problem 12 (chained and modified HS47) in the paper

    L. Luksan
    Hybrid methods in large sparse nonlinear least squares
    J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.

    classification NOR2-AN-V-V
    """

    s: int = 32  # Seed for dimensions
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, s: int = 32):
        self.s = s

    @property
    def n(self) -> int:
        """Number of variables: 3 * S + 2."""
        return 3 * self.s + 2

    def num_residuals(self) -> int:
        """Number of residuals: 6 * S."""
        return 6 * self.s

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.full(self.n, -1.0, dtype=jnp.float64)

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector."""
        x = y
        s = self.s
        residuals = []

        # Process S blocks, each contributing 6 equations
        i = 0  # Variable index (0-based)
        for j in range(s):
            # Extract relevant variables for this block
            x0 = x[i]
            x1 = x[i + 1]
            x2 = x[i + 2]
            x3 = x[i + 3]
            x4 = x[i + 4]

            # E(k): 10*x0^2 - 10*x1
            res1 = 10.0 * x0**2 - 10.0 * x1
            residuals.append(res1)

            # E(k+1): x2 - 1.0 (constant RHS = 1.0)
            res2 = x2 - 1.0
            residuals.append(res2)

            # E(k+2): (x3 - 1)^2
            res3 = (x3 - 1.0) ** 2
            residuals.append(res3)

            # E(k+3): (x4 - 1)^3
            res4 = (x4 - 1.0) ** 3
            residuals.append(res4)

            # E(k+4): x3*x0^2 + sin(x3-x4) - 10.0
            res5 = x3 * x0**2 + jnp.sin(x3 - x4) - 10.0
            residuals.append(res5)

            # E(k+5): x2*x3 + x1 - 20.0
            res6 = x2 * x3 + x1 - 20.0
            residuals.append(res6)

            # Move to next block (3 variables per block)
            i += 3

        return jnp.array(residuals, dtype=jnp.float64)

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
