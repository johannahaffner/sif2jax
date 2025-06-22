import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DENSCHNE(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem E.

    This is a 3-dimensional unconstrained optimization problem with
    squares of variables and an exponential term.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 3  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # From AMPL model: x[1]^2 + (x[2]+x[2]^2)^2 + (-1+exp(x[3]))^2
        term1 = x1**2
        term2 = (x2 + x2**2) ** 2
        term3 = (-1.0 + jnp.exp(x3)) ** 2

        return term1 + term2 + term3

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([2.0, 3.0, -8.0])

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at x1=0, x2=0, x3=0 (where exp(0)=1)
        return jnp.array([0.0, 0.0, 0.0])

    def expected_objective_value(self):
        # At x = [0, 0, 0]: 0^2 + (0+0^2)^2 + (-1+exp(0))^2 = 0 + 0 + 0 = 0
        return jnp.array(0.0)
