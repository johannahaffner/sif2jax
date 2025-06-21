import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class PENALTY1(AbstractUnconstrainedMinimisation):
    """
    PENALTY1 problem.

    This problem is a sum of n+1 least-squares groups, the first n of
    which have only a linear element.
    It Hessian matrix is dense.

    Source:  Problem 23 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley #181 (p. 79).

    SIF input: Ph. Toint, Dec 1989.

    classification SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 1000

    def objective(self, y, args):
        del args

        # First n groups: 100000 * (x[i] - 1)^2 for i=0 to n-1
        # Each group G(i) has x[i] with coefficient 1.0, constant -1.0, and scale 100000
        # Use float64 to ensure precision
        y = jnp.asarray(y, dtype=jnp.float64)
        linear_terms = y - 1.0
        scaled_squared = jnp.float64(100000.0) * (linear_terms * linear_terms)
        sum1 = jnp.sum(scaled_squared, dtype=jnp.float64)

        # Last group G(M=n+1): (sum of x[i]^2 - 0.25)^2
        # This group has elements E(i) which are x[i]^2, and constant -0.25
        x_squared = y * y
        elements_sum = jnp.sum(x_squared, dtype=jnp.float64)
        group_val = elements_sum - jnp.float64(0.25)
        sum2 = group_val * group_val

        return sum1 + sum2

    def y0(self):
        # Starting point: X(I) = I for I=1 to N
        return inexact_asarray(jnp.arange(1.0, self.n + 1.0))

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in detail in SIF file
        return None

    def expected_objective_value(self):
        # Solution values provided for some values of n
        n = self.n
        if n == 4:
            return jnp.array(2.24997e-4)
        elif n == 10:
            return jnp.array(7.08765e-5)
        else:
            # For other values, solution not provided
            return None

    def num_variables(self):
        return self.n
