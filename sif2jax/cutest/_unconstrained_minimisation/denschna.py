import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DENSCHNA(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem A.

    This is a 2-dimensional unconstrained optimization problem with
    nonlinear terms including exponentials.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # From AMPL model: x[1]^4 + (x[1]+x[2])^2 + (-1.0+exp(x[2]))^2
        # Compute powers and exponential once
        x1_sq = x1 * x1
        x1_4 = x1_sq * x1_sq
        exp_x2 = jnp.exp(x2)
        x1_plus_x2 = x1 + x2

        term1 = x1_4
        term2 = x1_plus_x2 * x1_plus_x2
        term3 = (-1.0 + exp_x2) ** 2

        return term1 + term2 + term3

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at x = [0.0, 0.0]
        return jnp.array([0.0, 0.0])

    def expected_objective_value(self):
        # At x = [0.0, 0.0]: 0^4 + (0+0)^2 + (-1+exp(0))^2 = 0 + 0 + 0 = 0
        return jnp.array(0.0)
