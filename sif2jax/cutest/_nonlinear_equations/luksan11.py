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

        # Vectorized computation
        # Extract x[0:s] and x[1:s+1] for vectorized operations
        x_i = x[:-1]  # x[0] to x[s-1]
        x_ip1 = x[1:]  # x[1] to x[s]

        # First residuals: 20*x[i]/(1+x[i]^2) - 10*x[i+1]
        d = 1.0 + x_i * x_i
        res1 = 20.0 * x_i / d - 10.0 * x_ip1

        # Second residuals: x[i] with RHS adjustment
        res2 = x_i.copy()

        # Create residuals array by interleaving res1 and res2
        residuals = jnp.zeros(2 * s, dtype=jnp.float64)
        residuals = residuals.at[::2].set(res1)  # Even indices: 0, 2, 4, ...
        residuals = residuals.at[1::2].set(res2)  # Odd indices: 1, 3, 5, ...

        # Apply constants: subtract 1.0 from odd indices (1, 3, 5, ...)
        # This corresponds to RHS = 1.0 for alternating equations
        odd_mask = jnp.arange(2 * s) % 2 == 1
        residuals = residuals - odd_mask.astype(jnp.float64)

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
