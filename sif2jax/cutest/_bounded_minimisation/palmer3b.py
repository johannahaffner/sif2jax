"""
A nonlinear least squares problem with bounds arising from chemical kinetics.

model: H-N=C=S TZVP + MP2
fitting Y to A2 X**2 + A4 X**4 + B / ( C + X**2 ), B, C nonnegative.

Source:
M. Palmer, Edinburgh, private communication.

SIF input: Nick Gould, 1990.

classification SBR2-RN-4-0
"""

import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PALMER3B(AbstractBoundedMinimisation):
    @property
    def name(self) -> str:
        return "PALMER3B"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 4  # 4 variables: A2, A4, B, C
    m: int = 23  # 23 data points

    @property
    def y0(self):
        # All variables start at 1.0
        return jnp.array([1.0, 1.0, 1.0, 1.0])

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
        a2, a4, b, c = y
        x_data, y_data = args

        # Model: Y = A2 * X^2 + A4 * X^4 + B / (C + X^2)
        x_sqr = x_data**2
        x_quart = x_sqr**2

        predicted = a2 * x_sqr + a4 * x_quart + b / (c + x_sqr)

        # Compute sum of squared residuals
        residuals = predicted - y_data
        return jnp.array(jnp.sum(residuals**2))

    @property
    def bounds(self):
        # A2 and A4 are free (unbounded)
        # B and C have lower bound 0.00001
        lower = jnp.array([-jnp.inf, -jnp.inf, 0.00001, 0.00001])
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf])
        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # According to the SIF file comment (line 151),
        # the optimal objective value is 4.227647
        return jnp.array(4.227647)
