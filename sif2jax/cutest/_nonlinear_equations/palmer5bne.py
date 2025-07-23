"""
A nonlinear least squares problem with bounds arising from chemical kinetics.

Model: H-N=C=Se TZVP + MP2
Fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 + A10 X**10
            + A12 X**12 + B / ( C + X**2 ), B, C nonnegative.

Source:
M. Palmer, Edinburgh, private communication.

SIF input: Nick Gould, 1992.
Bound-constrained nonlinear equations version: Nick Gould, June 2019.

classification NOR2-RN-9-12
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class PALMER5BNE(AbstractNonlinearEquations):
    @property
    def name(self) -> str:
        return "PALMER5BNE"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    m: int = 12  # Number of data points/equations (from 12 to 23)
    n: int = 9  # Number of variables

    @property
    def y0(self):
        return jnp.ones(self.n)

    @property
    def args(self):
        # X data (radians) - only indices 12-23 are used
        x_data = jnp.array(
            [
                0.000000,  # X12
                1.570796,  # X13
                1.396263,  # X14
                1.308997,  # X15
                1.221730,  # X16
                1.125835,  # X17
                1.047198,  # X18
                0.872665,  # X19
                0.698132,  # X20
                0.523599,  # X21
                0.349066,  # X22
                0.174533,  # X23
            ]
        )

        # Y data (KJmol-1)
        y_data = jnp.array(
            [
                83.57418,  # Y12
                81.007654,  # Y13
                18.983286,  # Y14
                8.051067,  # Y15
                2.044762,  # Y16
                0.000000,  # Y17
                1.170451,  # Y18
                10.479881,  # Y19
                25.785001,  # Y20
                44.126844,  # Y21
                62.822177,  # Y22
                77.719674,  # Y23
            ]
        )

        return x_data, y_data

    def residual(self, y, args):
        """Compute residuals for the nonlinear equations."""
        a0, a2, a4, a6, a8, a10, a12, b, c = y
        x_data, y_data = args

        # Compute powers of x
        x_sqr = x_data * x_data
        x_quart = x_sqr * x_sqr
        x_sext = x_quart * x_sqr
        x_8 = x_sqr * x_sext
        x_10 = x_sqr * x_8
        x_12 = x_sqr * x_10

        # Compute model predictions
        polynomial = (
            a0
            + a2 * x_sqr
            + a4 * x_quart
            + a6 * x_sext
            + a8 * x_8
            + a10 * x_10
            + a12 * x_12
        )
        rational = b / (c + x_sqr)
        predictions = polynomial + rational

        # Return residuals (predictions - y_data)
        return predictions - y_data

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self):
        # B and C have lower bounds of 0.00001
        lower = jnp.array(
            [
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                0.00001,
                0.00001,
            ]
        )
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment: 4.0606141D-02
        return jnp.array(4.0606141e-02)
