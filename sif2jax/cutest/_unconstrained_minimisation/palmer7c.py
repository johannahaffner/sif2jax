import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class PALMER7C(AbstractUnconstrainedMinimisation):
    """A linear least squares problem arising from chemical kinetics.

    model: H-N=C=Se TZVP + MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
                 A10 X**10 + A12 X**12 + A14 X**14 + A16 X**16

    Source:
    M. Palmer, Edinburgh, private communication.

    SIF input: Nick Gould, 1990.

    classification: QUR2-RN-8-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 8  # 8 coefficients
    m: int = 13  # 13 data points (X12-X24)

    @property
    def y0(self):
        # All coefficients start at 1.0
        return jnp.ones(self.n)

    @property
    def args(self):
        # X data values (radians) - only X12-X24
        x_data = jnp.array(
            [
                0.000000,
                0.139626,
                0.261799,
                0.436332,
                0.565245,
                0.512942,
                0.610865,
                0.785398,
                0.959931,
                1.134464,
                1.308997,
                1.483530,
                1.658063,
            ]
        )

        # Y data values (KJmol-1)
        y_data = jnp.array(
            [
                4.419446,
                3.564931,
                2.139067,
                0.404686,
                0.000000,
                0.035152,
                0.146813,
                2.718058,
                9.474417,
                26.132221,
                41.451561,
                72.283164,
                117.630959,
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
        return jnp.array(5.0310687e-02)
