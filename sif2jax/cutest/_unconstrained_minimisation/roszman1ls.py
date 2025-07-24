"""
NIST Data fitting problem ROSZMAN1.

Fit: y = b1 - b2*x - arctan[b3/(x-b4)]/pi + e

Source:
Problem from the NIST nonlinear regression test set
http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

Reference:
Roszman, L., NIST (1979).
Quantum Defects for Sulfur I Atom.

SIF input: Nick Gould and Tyrone Rees, Oct 2015

classification SUR2-MN-4-0

"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class ROSZMAN1LS(AbstractUnconstrainedMinimisation):
    @property
    def name(self) -> str:
        return "ROSZMAN1LS"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    n: int = 4  # 4 parameters
    m: int = 25  # 25 data points

    @property
    def y0(self):
        if self.y0_iD == 0:
            # START1
            return jnp.array([0.1, -0.00001, 1000.0, -100.0])
        else:
            # START2
            return jnp.array([0.2, -0.000005, 1200.0, -150.0])

    @property
    def args(self):
        # X data
        x_data = jnp.array(
            [
                -4868.68,
                -4868.09,
                -4867.41,
                -3375.19,
                -3373.14,
                -3372.03,
                -2473.74,
                -2472.35,
                -2469.45,
                -1894.65,
                -1893.40,
                -1497.24,
                -1495.85,
                -1493.41,
                -1208.68,
                -1206.18,
                -1206.04,
                -997.92,
                -996.61,
                -996.31,
                -834.94,
                -834.66,
                -710.03,
                -530.16,
                -464.17,
            ]
        )

        # Y data
        y_data = jnp.array(
            [
                0.252429,
                0.252141,
                0.251809,
                0.297989,
                0.296257,
                0.295319,
                0.339603,
                0.337731,
                0.333820,
                0.389510,
                0.386998,
                0.438864,
                0.434887,
                0.427893,
                0.471568,
                0.461699,
                0.461144,
                0.513532,
                0.506641,
                0.505062,
                0.535648,
                0.533726,
                0.568064,
                0.612886,
                0.624169,
            ]
        )

        return (x_data, y_data)

    def objective(self, y, args):
        b1, b2, b3, b4 = y
        x_data, y_data = args

        # Model: y = b1 - b2*x - arctan[b3/(x-b4)]/pi
        # Note: The SIF file element E7 computes -ATAN(V1/(V2-X))/PI
        # where V1=B3, V2=B4, X=x_data
        # This means the term is -arctan(b3/(b4-x))/pi, not -arctan(b3/(x-b4))/pi
        # However, since arctan is an odd function: arctan(-z) = -arctan(z)
        # So -arctan(b3/(b4-x))/pi = -arctan(-b3/(x-b4))/pi = arctan(b3/(x-b4))/pi
        model_vals = b1 - b2 * x_data + jnp.arctan(b3 / (x_data - b4)) / jnp.pi
        residuals = model_vals - y_data

        # Sum of squared residuals
        return jnp.sum(residuals**2)

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # The optimal objective value is not explicitly given in the SIF file
        return None
