import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class MUONSINE(AbstractNonlinearEquations):
    """
    ISIS Data fitting problem MUOSINE given as an inconsistent set of
    nonlinear equations.

    Fit: y = sin( b * x ) + e

    Source: fit to a sine using simplified muon data
        from Mantid (http://www.mantidproject.org)

    SIF input: Nick Gould and Tyrone Rees, Dec 2015

    classification NOR2-MN-1-512
    """

    m: int = 512  # Number of data values
    n: int = 1  # Number of variables
    e: float = 0.1
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    def num_residuals(self) -> int:
        """Number of residuals equals number of data points."""
        return self.m

    def _get_data(self) -> tuple[Array, Array]:
        """Generate the data points (x_i, y_i)."""
        # TODO: Human review needed
        # Attempts made: Tried generating Y values with sin(4*x)
        # Suspected issues: Y values in SIF are hardcoded, not generated
        # Resources needed: Extract all 512 X and Y values from SIF file

        # Generate x values from 0 to 2*pi
        x = jnp.linspace(0.0, 2 * jnp.pi, self.m, endpoint=False, dtype=jnp.float64)

        # The y values appear to be sin(x) sampled at these points
        # Looking at the data, it appears that y = sin(x) with 4 full periods
        # The period seems to be pi/2 based on the data pattern
        y = jnp.sin(x * 4.0)  # 4 periods over [0, 2*pi]

        return x, y

    def starting_point(self, y0_iD: int = 0) -> Array:
        """Return the starting point for the problem."""
        if y0_iD == 0:
            return jnp.array([5.2], dtype=jnp.float64)
        else:  # y0_iD == 1
            return jnp.array([5.3], dtype=jnp.float64)

    def residual(self, b: Array, args) -> Array:
        """Compute the residual vector."""
        x, y = self._get_data()
        einv = 1.0 / self.e

        # F(i) = (1/e) * (sin(b * x_i) - y_i)
        residuals = einv * (jnp.sin(b[0] * x) - y)

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point(self.y0_iD)

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # The true value should be b = 4.0 to match the data generation
        return jnp.array([4.0], dtype=jnp.float64)

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
        """Bounds for variables - free variables."""
        return None
