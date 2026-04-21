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

        x0 = y[: n - 4]
        x1 = y[1 : n - 3]
        x2 = y[2 : n - 2]
        x3 = y[3 : n - 1]
        xn = y[n - 1]

        r_lin = -4 * x0 + 3.0
        r_quad = x0**2 + 2 * x1**2 + 3 * x2**2 + 4 * x3**2 + 5 * xn**2
        r = jnp.concatenate([r_lin, r_quad])
        return jnp.sum(r**2)

    @property
    def y0(self):
        # Initial values from SIF file (all 1.0)
        return inexact_asarray(jnp.ones(self.n))

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    @property
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
