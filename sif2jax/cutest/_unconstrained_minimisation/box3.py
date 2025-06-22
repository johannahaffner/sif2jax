import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BOX3(AbstractUnconstrainedMinimisation):
    """Box problem in 3 variables.

    This function is a nonlinear least squares with 10 groups. Each
    group has 2 nonlinear elements of exponential type.

    Source: Problem 12 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#BOX663
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 3  # Problem has 3 variables
    m: int = 10  # Number of data points

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # Define indices from 1 to m
        indices = jnp.arange(1, self.m + 1)

        # Define inner function to compute residual for a single index i
        def compute_residual(i):
            i_float = inexact_asarray(i)
            t_i = -0.1 * i_float

            # Compute exp(-0.1*i*x1) - exp(-0.1*i*x2)
            term1 = jnp.exp(t_i * x1) - jnp.exp(t_i * x2)

            # Compute coefficient: -exp(-0.1*i) + exp(-i)
            coeff = -jnp.exp(t_i) + jnp.exp(-i_float)

            # Compute the residual: x3 * coeff + term1
            residual = x3 * coeff + term1

            return residual**2

        # Vectorize the function over indices and sum the results
        residuals = jax.vmap(compute_residual)(indices)
        return jnp.sum(residuals)

    def y0(self):
        # Initial values from SIF file
        return inexact_asarray(jnp.array([0.0, 10.0, 1.0]))

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is approximately (1, 10, 1)
        return jnp.array([1.0, 10.0, 1.0])

    def expected_objective_value(self):
        # According to the SIF file comment (line 104),
        # the optimal objective value is 0.0
        return jnp.array(0.0)
