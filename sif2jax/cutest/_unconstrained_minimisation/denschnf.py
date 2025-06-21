import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DENSCHNF(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem F.

    This is a 2-dimensional unconstrained optimization problem with
    a sum of squares formulation.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AY-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # From AMPL model:
        # (2*(x1+x2)^2+(x1-x2)^2-8)^2 + (5*x1^2+(x2-3)^2-9)^2
        term1 = (2.0 * (x1 + x2) ** 2 + (x1 - x2) ** 2 - 8.0) ** 2
        term2 = (5.0 * x1**2 + (x2 - 3.0) ** 2 - 9.0) ** 2

        return term1 + term2

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([2.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # Solution depends on solving the system of equations
        return None

    def expected_objective_value(self):
        # The minimum value is 0 when both terms equal 0
        return jnp.array(0.0)
