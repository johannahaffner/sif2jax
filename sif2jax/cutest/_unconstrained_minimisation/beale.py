import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BEALE(AbstractUnconstrainedMinimisation):
    """Beale function in 2 variables.

    Source: Problem 5 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#89.
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Problem has 2 variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # Beale function from AMPL:
        # (-1.5+x[1]*(1.0-x[2]))^2 + (-2.25+x[1]*(1.0-x[2]^2))^2 +
        # (-2.625+x[1]*(1.0-x[2]^3))^2

        # First term: (-1.5 + x1*(1-x2))^2
        term1 = (-1.5 + x1 * (1.0 - x2)) ** 2

        # Second term: (-2.25 + x1*(1-x2^2))^2
        term2 = (-2.25 + x1 * (1.0 - x2**2)) ** 2

        # Third term: (-2.625 + x1*(1-x2^3))^2
        term3 = (-2.625 + x1 * (1.0 - x2**3)) ** 2

        return term1 + term2 + term3

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return jnp.array([1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is known to be (3, 0.5)
        return jnp.array([3.0, 0.5])

    def expected_objective_value(self):
        # According to the SIF file comment (line 91),
        # the optimal objective value is 0.0
        return jnp.array(0.0)
