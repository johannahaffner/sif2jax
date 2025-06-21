import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DENSCHND(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem D.

    This is a 3-dimensional unconstrained optimization problem with
    polynomial terms up to the fourth power.

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

        # From AMPL model:
        # (x[1]^2+x[2]^3-x[3]^4)^2 + (2*x[1]*x[2]*x[3])^2 +
        # (2*x[1]*x[2]-3*x[2]*x[3]+x[1]*x[3])^2
        term1 = (x1**2 + x2**3 - x3**4) ** 2
        term2 = (2.0 * x1 * x2 * x3) ** 2
        term3 = (2.0 * x1 * x2 - 3.0 * x2 * x3 + x1 * x3) ** 2

        return term1 + term2 + term3

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([10.0, 10.0, 10.0])

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.array([0.0, 0.0, 0.0])

    def expected_objective_value(self):
        # At the origin, all terms evaluate to 0
        return jnp.array(0.0)
