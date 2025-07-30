import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed
# Attempts made: Vectorized implementation
# Suspected issues: Jacobian mismatch in last constraint row
# Resources needed: Detailed comparison of constraint derivatives
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

        # Vectorized approach
        # Create indices for all blocks
        i_indices = jnp.arange(s) * 3  # [0, 3, 6, ..., 3*(s-1)]

        # Extract variables for all blocks
        x0 = x[i_indices]  # X(I) in SIF
        x1 = x[i_indices + 1]  # X(I+1) in SIF
        x2 = x[i_indices + 2]  # X(I+2) in SIF
        x3 = x[i_indices + 3]  # X(I+3) in SIF
        x4 = x[i_indices + 4]  # X(I+4) in SIF

        # Compute all residuals for each type
        # E(k): 10*x0^2 - 10*x1
        res1 = 10.0 * x0**2 - 10.0 * x1

        # E(k+1): x2 - 1.0
        res2 = x2 - 1.0

        # E(k+2): (x3 - 1)^2
        res3 = (x3 - 1.0) ** 2

        # E(k+3): (x4 - 1)^3
        res4 = (x4 - 1.0) ** 3

        # E(k+4): x3*x0^2 + sin(x3-x4) - 10.0
        res5 = x3 * x0**2 + jnp.sin(x3 - x4) - 10.0

        # E(k+5): x2*x3 + x1 - 20.0
        res6 = x2 * x3 + x1 - 20.0

        # Stack residuals in the correct order
        # Each block contributes 6 residuals in sequence
        residuals = jnp.stack([res1, res2, res3, res4, res5, res6], axis=1).flatten()

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
