import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class PALMER1ANE(AbstractNonlinearEquations):
    """
    A nonlinear least squares problem with bounds
    arising from chemical kinetics.

    model: H-N=N=N TZVP+MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6
                 + B / ( C + X**2 ), B, C nonnegative.

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1990.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-RN-6-35
    """

    n: int = 6
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # No custom __init__ needed - using Equinox module defaults

    def num_residuals(self) -> int:
        """Number of residuals equals number of data points."""
        return 35

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.ones(self.n, dtype=jnp.float64)

    def _get_data(self):
        """Get the data points for the problem."""
        # X values (radians)
        x = jnp.array(
            [
                -1.788963,
                -1.745329,
                -1.658063,
                -1.570796,
                -1.483530,
                -1.396263,
                -1.308997,
                -1.218612,
                -1.134464,
                -1.047198,
                -0.872665,
                -0.698132,
                -0.523599,
                -0.349066,
                -0.174533,
                0.0000000,
                1.788963,
                1.745329,
                1.658063,
                1.570796,
                1.483530,
                1.396263,
                1.308997,
                1.218612,
                1.134464,
                1.047198,
                0.872665,
                0.698132,
                0.523599,
                0.349066,
                0.174533,
                -1.8762289,
                -1.8325957,
                1.8762289,
                1.8325957,
            ],
            dtype=jnp.float64,
        )

        # Y values (KJmol-1)
        y = jnp.array(
            [
                78.596218,
                65.77963,
                43.96947,
                27.038816,
                14.6126,
                6.2614,
                1.538330,
                0.000000,
                1.188045,
                4.6841,
                16.9321,
                33.6988,
                52.3664,
                70.1630,
                83.4221,
                88.3995,
                78.596218,
                65.77963,
                43.96947,
                27.038816,
                14.6126,
                6.2614,
                1.538330,
                0.000000,
                1.188045,
                4.6841,
                16.9321,
                33.6988,
                52.3664,
                70.1630,
                83.4221,
                108.18086,
                92.733676,
                108.18086,
                92.733676,
            ],
            dtype=jnp.float64,
        )

        return x, y

    def residual(self, v: Array, args) -> Array:
        """Compute the residual vector."""
        # Variables: A0, A2, A4, A6, B, C
        a0, a2, a4, a6, b, c = v

        x, y = self._get_data()

        # Compute powers of x
        x_sqr = x * x
        x_quart = x_sqr * x_sqr
        x_sext = x_quart * x_sqr

        # Model: A0 + A2*X^2 + A4*X^4 + A6*X^6 + B/(C + X^2)
        model = a0 + a2 * x_sqr + a4 * x_quart + a6 * x_sext + b / (c + x_sqr)

        # Residuals
        residuals = model - y

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
        # Solution should satisfy F(x*) = 0
        return jnp.zeros(self.n, dtype=jnp.float64)

    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """Bounds for variables.

        A0, A2, A4, A6 are free
        B >= 0.00001
        C >= 0.00001
        """
        lower = jnp.array(
            [-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.00001, 0.00001],
            dtype=jnp.float64,
        )
        upper = jnp.full(self.n, jnp.inf, dtype=jnp.float64)
        return lower, upper
