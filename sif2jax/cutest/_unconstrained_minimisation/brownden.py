import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BROWNDEN(AbstractUnconstrainedMinimisation):
    """Brown and Dennis function.

    Brown and Dennis problem in 4 variables.
    This function is a nonlinear least squares with 20 groups. Each
    group has 2 nonlinear elements.

    Source: Problem 16 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#30
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-4-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 4  # Problem has 4 variables
    m: int = 20  # Number of groups

    def objective(self, y, args):
        del args
        x1, x2, x3, x4 = y

        # Compute the objective function according to SIF specification
        # For each i from 1 to 20:
        # t_i = i/5
        # A_i = x1 + t_i * x2 - exp(t_i)
        # B_i = x3 + x4 * sin(t_i) - cos(t_i)
        # G_i = A_i^2 + B_i^2
        # objective = sum(G_i^2)

        # Vectorized computation for better performance
        i_values = inexact_asarray(jnp.arange(1, self.m + 1))
        t_values = i_values / 5.0

        # Element A(i): x1 + t_i * x2 - exp(t_i)
        A_values = x1 + t_values * x2 - jnp.exp(t_values)

        # Element B(i): x3 + x4 * sin(t_i) - cos(t_i)
        B_values = x3 + x4 * jnp.sin(t_values) - jnp.cos(t_values)

        # Group G(i) = A_i^2 + B_i^2, then square it
        G_values = A_values**2 + B_values**2

        return jnp.sum(G_values**2)

    def y0(self):
        # Initial values from SIF file
        return inexact_asarray(jnp.array([25.0, 5.0, -5.0, -1.0]))

    def args(self):
        return None

    def expected_result(self):
        # No specific expected result provided in SIF
        return None

    def expected_objective_value(self):
        # According to SIF comment, solution value is approximately 85822.2
        return jnp.array(85822.2)
