import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class HS1NE(AbstractNonlinearEquations):
    """
    The ever famous 2 variables Rosenbrock "banana valley" problem
    with a single lower bound.

    Source: problem 1 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: A.R. Conn, March 1990.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-AN-2-2
    """

    m: int = 2
    n: int = 2
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.array([-2.0, 1.0], dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the HS1 problem"""
        x1, x2 = y[0], y[1]

        # Initialize residuals
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        # G1: 0.1 * (x2 - x1^2) = 0
        # Element E1: -x1^2, scale 0.1
        # Note: pycutest inverts the scale factor for NLE problems
        residuals = residuals.at[0].set(10.0 * (x2 - x1 * x1))

        # G2: x1 - 1.0 = 0
        residuals = residuals.at[1].set(x1 - 1.0)

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
        # Solution would be x1=1, x2=1
        return jnp.array([1.0, 1.0], dtype=jnp.float64)

    @property
    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array]:
        """Lower bound on x2 from the SIF file."""
        # LO HS1 X2 -1.5
        lower = jnp.array([-jnp.inf, -1.5], dtype=jnp.float64)
        upper = jnp.inf * jnp.ones_like(lower)
        return lower, upper
