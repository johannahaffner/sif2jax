"""
A nonlinear least squares problem arising from chemical kinetics.

Model: H-N=C=Se TZVP + MP2

Fitting Y to:
    A0 + A2 X^2 + A4 X^4 + A6 X^6 + A8 X^8 + A10 X^10 + L * exp(-K X^2)

This is a nonlinear equations reformulation where the residuals are set to zero,
making it a system of 23 nonlinear equations in 8 variables.

Source:
M. Palmer, Edinburgh, private communication.

SIF input: Nick Gould, 1990.
Bound-constrained nonlinear equations version: Nick Gould, June 2019.

classification NOR2-RN-8-23
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class PALMER4ENE(AbstractNonlinearEquations):
    @property
    def name(self) -> str:
        return "PALMER4ENE"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 8  # 8 variables: A0, A2, A4, A6, A8, A10, K, L
    m: int = 23  # 23 equations

    @property
    def y0(self):
        # All variables start at 1.0
        return jnp.ones(self.n)

    @property
    def args(self):
        # X values (radians)
        x = jnp.array(
            [
                -1.658063,
                -1.570796,
                -1.396263,
                -1.221730,
                -1.047198,
                -0.872665,
                -0.741119,
                -0.698132,
                -0.523599,
                -0.349066,
                -0.174533,
                0.0,
                0.174533,
                0.349066,
                0.523599,
                0.698132,
                0.741119,
                0.872665,
                1.047198,
                1.221730,
                1.396263,
                1.570796,
                1.658063,
            ]
        )

        # Y values (KJmol-1)
        y_data = jnp.array(
            [
                67.27625,
                52.8537,
                30.2718,
                14.9888,
                5.5675,
                0.92603,
                0.0,
                0.085108,
                1.867422,
                5.014768,
                8.263520,
                9.8046208,
                8.263520,
                5.014768,
                1.867422,
                0.085108,
                0.0,
                0.92603,
                5.5675,
                14.9888,
                30.2718,
                52.8537,
                67.27625,
            ]
        )

        return (x, y_data)

    def residual(self, y, args):
        x, y_data = args

        # Variables
        a0, a2, a4, a6, a8, a10, k, l = y

        # Compute powers of x
        x2 = x * x
        x4 = x2 * x2
        x6 = x2 * x4
        x8 = x2 * x6
        x10 = x2 * x8

        # Model prediction
        polynomial = a0 + a2 * x2 + a4 * x4 + a6 * x6 + a8 * x8 + a10 * x10
        exponential = l * jnp.exp(-k * x2)
        y_pred = polynomial + exponential

        # Residuals (equations set to zero)
        residuals = y_pred - y_data

        return residuals

    @property
    def bounds(self):
        # A0, A2, A4, A6, A8, A10, and L are free (FR)
        # K has default bounds (lower bound 0)
        lower = jnp.array(
            [-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, 0.0, -jnp.inf]
        )
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def expected_objective_value(self):
        # For nonlinear equations, the objective is always zero
        return jnp.array(0.0)

    @property
    def expected_residual_norm(self):
        # The SIF file mentions: SOLTN 1.48003482D-04
        # This appears to be the optimal objective value for the least squares problem
        # For the equations version, the residual norm at the solution should be
        # close to sqrt(2 * objective)
        return jnp.array(jnp.sqrt(2 * 1.48003482e-04))
