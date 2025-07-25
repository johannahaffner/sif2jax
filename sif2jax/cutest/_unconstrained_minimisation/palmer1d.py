import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class PALMER1D(AbstractUnconstrainedMinimisation):
    """A linear least squares problem arising from chemical kinetics.

    model: H-N=N=N TZVP+MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
                 A10 X**10 + A12 X**12

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1990.

    classification: QUR2-RN-7-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 7  # 7 coefficients: A0, A2, A4, A6, A8, A10, A12
    m: int = 35  # 35 data points

    @property
    def y0(self):
        # All coefficients start at 1.0
        return jnp.ones(self.n)

    @property
    def args(self):
        # X data values (radians) - same as PALMER1C
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
                -1.8762289,
                -1.8325957,
                1.8762289,
                1.8325957,
            ]
        )

        # Y data values (KJmol-1) - same as PALMER1C
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
                108.18086,
                92.733676,
                108.18086,
                92.733676,
            ]
        )

        return (x_data, y_data)

    def objective(self, y, args):
        """Compute the sum of squared residuals."""
        a0, a2, a4, a6, a8, a10, a12 = y
        x_data, y_data = args

        # Vectorized computation of powers
        x_sqr = x_data**2
        x_quart = x_sqr**2
        x_6 = x_sqr * x_quart
        x_8 = x_sqr * x_6
        x_10 = x_sqr * x_8
        x_12 = x_sqr * x_10

        # Model prediction
        predicted = (
            a0
            + a2 * x_sqr
            + a4 * x_quart
            + a6 * x_6
            + a8 * x_8
            + a10 * x_10
            + a12 * x_12
        )

        # Compute sum of squared residuals
        residuals = predicted - y_data
        return jnp.sum(residuals**2)

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment
        return jnp.array(0.652673985)
