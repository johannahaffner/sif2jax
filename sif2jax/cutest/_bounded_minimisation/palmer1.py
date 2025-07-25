import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PALMER1(AbstractBoundedMinimisation):
    """A nonlinear least squares problem with bounds arising from chemical kinetics.

    model: H-N=N=N TZVP+MP2
    fitting Y to A X**2 + B / ( C + X**2 / D ), B, C, D nonnegative.

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1990.

    classification: SBR2-RN-4-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 4  # 4 variables: A, B, C, D
    m: int = 31  # 31 data points

    @property
    def y0(self):
        # All variables start at 1.0
        return jnp.ones(self.n)

    @property
    def args(self):
        # X data values (radians)
        x_data = jnp.array(
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
            ]
        )

        # Y data values (KJmol-1)
        y_data = jnp.array(
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
            ]
        )

        return (x_data, y_data)

    def objective(self, y, args):
        """Compute the sum of squared residuals."""
        a, b, c, d = y
        x_data, y_data = args

        # Model: Y = A * X^2 + B / (C + X^2 / D)
        x_sqr = x_data**2

        predicted = a * x_sqr + b / (c + x_sqr / d)

        # Compute sum of squared residuals
        residuals = predicted - y_data
        return jnp.sum(residuals**2)

    @property
    def bounds(self):
        # A is free (unbounded)
        # B, C, D have lower bound 0.00001
        lower = jnp.array([-jnp.inf, 0.00001, 0.00001, 0.00001])
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf])
        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment
        return jnp.array(11754.6025)
