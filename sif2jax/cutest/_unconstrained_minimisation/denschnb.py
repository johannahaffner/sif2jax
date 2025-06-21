import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DENSCHNB(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem B.

    This is a 2-dimensional unconstrained optimization problem with
    a product term (x1 - 2.0) * x2.

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

        # From AMPL model: (x[1]-2.0)^2 + ((x[1]-2.0)*x[2])^2 + (x[2]+1.0)^2
        term1 = (x1 - 2.0) ** 2
        term2 = ((x1 - 2.0) * x2) ** 2
        term3 = (x2 + 1.0) ** 2

        return term1 + term2 + term3

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # Based on the problem formulation, the minimum is at:
        return jnp.array([2.0, -1.0])

    def expected_objective_value(self):
        # At x = [2.0, -1.0]: (2-2)^2 + ((2-2)*(-1))^2 + (-1+1)^2 = 0
        return jnp.array(0.0)
