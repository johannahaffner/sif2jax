import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class HS25NE(AbstractNonlinearEquations):
    """A nonlinear least squares problem with bounds.

    Source: problem 25 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: J-M Collin, Mar 1990.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-AN-3-99
    """

    n: int = 3
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return 99

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.array([100.0, 12.5, 3.0])

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        Args:
            y: Array of shape (3,) containing [x1, x2, x3]
            args: Additional arguments (unused)

        Returns:
            Array of shape (99,) containing the residuals
        """
        x1, x2, x3 = y[0], y[1], y[2]

        # Create array of i values from 1 to 99
        i_values = jnp.arange(1, 100, dtype=jnp.float64)
        i_over_100 = i_values / 100.0

        # Compute ui values for each i
        log_term = jnp.log(i_over_100)
        exp_term = jnp.exp((2.0 / 3.0) * (-50.0 * log_term))
        ui = exp_term + 25.0

        # Compute the element function for each i
        xi = 1.0 / x1
        wmy = ui - x2
        wmyez = wmy**x3
        expo = jnp.exp(-xi * wmyez)

        # Compute residuals
        residuals = expo - i_over_100

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Solution from AMPL model shows approximately:
        # x1 = 50, x2 = 25, x3 = 1.5
        return jnp.array([50.0, 25.0, 1.5])

    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array]:
        """Return the bounds for the variables."""
        lower = jnp.array([0.1, 0.0, 0.0])
        upper = jnp.array([100.0, 25.6, 5.0])
        return lower, upper
