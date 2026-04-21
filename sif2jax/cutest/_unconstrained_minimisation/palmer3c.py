import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class PALMER3C(AbstractUnconstrainedMinimisation):
    """A linear least squares problem arising from chemical kinetics.

    model: H-N=C=S TZVP + MP2
    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
                 A10 X**10 + A12 X**12 + A14 X**14

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
                -1.658063,
                -1.570796,
                -1.396263,
                -1.221730,
                -1.047198,
                -0.872665,
                -0.766531,
                -0.698132,
                -0.523599,
                -0.349066,
                -0.174533,
                0.0,
                0.174533,
                0.349066,
                0.523599,
                0.698132,
                0.766531,
                0.872665,
                1.047198,
                1.221730,
                1.396263,
                1.570796,
                1.658063,
            ]
        )

        # Y data values (KJmol-1)
        y_data = jnp.array(
            [
                64.87939,
                50.46046,
                28.2034,
                13.4575,
                4.6547,
                0.59447,
                0.0000,
                0.2177,
                2.3029,
                5.5191,
                8.5519,
                9.8919,
                8.5519,
                5.5191,
                2.3029,
                0.2177,
                0.0000,
                0.59447,
                4.6547,
                13.4575,
                28.2034,
                50.46046,
                64.87939,
            ]
        )

        return (x_data, y_data)

    def objective(self, y, args):
        """Compute the sum of squared residuals."""
        x_data, y_data = args
        residuals = jnp.polyval(y[::-1], x_data**2) - y_data
        return jnp.sum(residuals**2)

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment
        return jnp.array(1.9537639e-02)
