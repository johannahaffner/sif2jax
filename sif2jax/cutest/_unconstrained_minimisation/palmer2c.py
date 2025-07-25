import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class PALMER2C(AbstractUnconstrainedMinimisation):
    """A linear least squares problem arising from chemical kinetics.

    model: H-N=C=O TZVP + MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
                 A10 X**10 + A12 X**12

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1990.

    classification: QUR2-RN-8-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 8  # 8 coefficients
    m: int = 23  # 23 data points

    @property
    def y0(self):
        # All coefficients start at 1.0
        return jnp.ones(self.n)

    @property
    def args(self):
        # X data values (radians)
        x_data = jnp.array(
            [
                -1.745329,
                -1.570796,
                -1.396263,
                -1.221730,
                -1.047198,
                -0.937187,
                -0.872665,
                -0.698132,
                -0.523599,
                -0.349066,
                -0.174533,
                0.0,
                0.174533,
                0.349066,
                0.523599,
                0.698132,
                0.872665,
                0.937187,
                1.047198,
                1.221730,
                1.396263,
                1.570796,
                1.745329,
            ]
        )

        # Y data values (KJmol-1)
        y_data = jnp.array(
            [
                72.676767,
                40.149455,
                18.8548,
                6.4762,
                0.8596,
                0.00000,
                0.2730,
                3.2043,
                8.1080,
                13.4291,
                17.7149,
                19.4529,
                17.7149,
                13.4291,
                8.1080,
                3.2053,
                0.2730,
                0.00000,
                0.8596,
                6.4762,
                18.8548,
                40.149455,
                72.676767,
            ]
        )

        return (x_data, y_data)

    def objective(self, y, args):
        """Compute the sum of squared residuals."""
        a0, a2, a4, a6, a8, a10, a12, a14 = y
        x_data, y_data = args

        # Vectorized computation of powers
        x_sqr = x_data**2
        x_quart = x_sqr**2
        x_6 = x_sqr * x_quart
        x_8 = x_sqr * x_6
        x_10 = x_sqr * x_8
        x_12 = x_sqr * x_10
        x_14 = x_sqr * x_12

        # Model prediction
        predicted = (
            a0
            + a2 * x_sqr
            + a4 * x_quart
            + a6 * x_6
            + a8 * x_8
            + a10 * x_10
            + a12 * x_12
            + a14 * x_14
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
        return jnp.array(1.4368886e-02)
