import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class BDQRTIC(AbstractUnconstrainedMinimisation):
    """BDQRTIC function.

    This problem is quartic and has a banded Hessian with bandwidth = 9.

    Source: Problem 61 in
    A.R. Conn, N.I.M. Gould, M. Lescrenier and Ph.L. Toint,
    "Performance of a multifrontal scheme for partially separable optimization",
    Report 88/4, Dept of Mathematics, FUNDP (Namur, B), 1988.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Other suggested values are 100, 500, and 1000

    def objective(self, y, args):
        del args
        n = self.n

        # Vectorized implementation for efficiency
        # For each i from 1 to n-4 (0 to n-5 in 0-based indexing)
        i = jnp.arange(n - 4)

        # First part: (-4*x[i] + 3.0)^2 for each group
        # From L(I) group with coefficient -4.0 and constant -3.0
        part1 = (-4 * y[i] + 3.0) ** 2

        # Second part: (x[i]^2 + 2*x[i+1]^2 + 3*x[i+2]^2 + 4*x[i+3]^2 + 5*x[N]^2)^2
        # From G(I) group with squared elements
        part2 = (
            y[i] ** 2
            + 2 * y[i + 1] ** 2
            + 3 * y[i + 2] ** 2
            + 4 * y[i + 3] ** 2
            + 5 * y[n - 1] ** 2
        ) ** 2

        return jnp.sum(part1 + part2)

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return inexact_asarray(jnp.ones(self.n))

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # Based on the SIF file comment for n=100 (line 103)
        if self.n == 100:
            return jnp.array(3.78769e02)
        elif self.n == 500:
            return jnp.array(1.98101e03)
        elif self.n == 1000:
            return jnp.array(3.98382e03)
        else:
            # For other values of n, the optimal value is not provided
            return None
