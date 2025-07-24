"""
A nonlinear least squares problem arising from chemical kinetics.

Model: H-N=C=S TZVP + MP2
Fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
             A10 X**10 + L * EXP( -K X**2 )

Source:
M. Palmer, Edinburgh, private communication.

SIF input: Nick Gould, 1990.
Bound-constrained nonlinear equations version: Nick Gould, June 2019.

classification NOR2-RN-8-23
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class PALMER3ENE(AbstractNonlinearEquations):
    @property
    def name(self) -> str:
        return "PALMER3ENE"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    m: int = 23  # Number of data points/equations
    n: int = 8  # Number of variables

    @property
    def y0(self):
        return jnp.ones(self.n)

    @property
    def args(self):
        # X data (radians)
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

        # Y data (KJmol-1)
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

        return x_data, y_data

    def residual(self, y, args):
        """Compute residuals for the nonlinear equations."""
        a0, a2, a4, a6, a8, a10, k, l = y
        x_data, y_data = args

        # Compute powers of x
        x_sqr = x_data * x_data
        x_quart = x_sqr * x_sqr
        x_6 = x_sqr * x_quart
        x_8 = x_sqr * x_6
        x_10 = x_sqr * x_8

        # Compute model predictions
        polynomial = a0 + a2 * x_sqr + a4 * x_quart + a6 * x_6 + a8 * x_8 + a10 * x_10
        exponential = l * jnp.exp(-k * x_sqr)
        predictions = polynomial + exponential

        # Return residuals (predictions - y_data)
        return predictions - y_data

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self):
        # K has default lower bound 0.0 in CUTEst
        lower = jnp.array(
            [-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.0, -jnp.inf]
        )
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment: 5.0741053D-05
        return jnp.array(5.0741053e-05)
