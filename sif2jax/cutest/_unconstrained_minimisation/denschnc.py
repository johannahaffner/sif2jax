import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DENSCHNC(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem C.

    This is a 2-dimensional unconstrained optimization problem with
    squares of variables and an exponential term.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # From AMPL model: (-2+x[1]^2+x[2]^2)^2 + (-2+exp(x[1]-1)+x[2]^3)^2
        term1 = (-2.0 + x1**2 + x2**2) ** 2
        term2 = (-2.0 + jnp.exp(x1 - 1.0) + x2**3) ** 2

        return term1 + term2

    def y0(self):
        # Initial values from AMPL model: x[1]=2, x[2]=3
        return jnp.array([2.0, 3.0])

    def args(self):
        return None

    def expected_result(self):
        # Solution depends on solving the system of equations
        return None

    def expected_objective_value(self):
        # The minimum value is 0 when both terms equal 0
        return jnp.array(0.0)
