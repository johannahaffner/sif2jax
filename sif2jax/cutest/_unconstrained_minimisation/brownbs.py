import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BROWNBS(AbstractUnconstrainedMinimisation):
    """Brown badly scaled function.

    Brown badly scaled problem in 2 variables.
    This problem is a sum of n-1 sets of 3 groups, one of them involving
    a nonlinear element and all being of the least square type.
    Its Hessian matrix is tridiagonal.

    Source: Problem 4 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#25
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Problem has 2 variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # From SIF file structure for n=2:
        # Group A(1): x1 with constant 1000000.0 -> (x1 - 1000000.0)^2
        # Group B(1): x2 with constant 0.000002 -> (x2 - 0.000002)^2
        # Group C(1): x1*x2 with constant 2.0 -> (x1*x2 - 2.0)^2

        term_a = (x1 - 1000000.0) ** 2
        term_b = (x2 - 0.000002) ** 2
        term_c = (x1 * x2 - 2.0) ** 2

        return term_a + term_b + term_c

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return jnp.array([1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution for this badly scaled problem
        return None

    def expected_objective_value(self):
        # The optimal objective value bound is 0.0
        return jnp.array(0.0)
