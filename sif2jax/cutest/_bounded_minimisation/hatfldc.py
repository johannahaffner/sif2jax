"""
A test problem from the OPTIMA user manual.

Source:
"The OPTIMA user manual (issue No.8, p. 26)",
Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

SIF input: Ph. Toint, May 1990.

classification SBR2-AN-25-0
"""

import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HATFLDC(AbstractBoundedMinimisation):
    @property
    def name(self) -> str:
        return "HATFLDC"

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 25  # 25 variables

    @property
    def y0(self):
        # All variables start at 0.9
        return jnp.full(self.n, 0.9)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        del args

        # The objective function has the structure:
        # G(1) = X(1) - 1.0
        # G(i) = X(i+1) - X(i)^2 for i = 2, ..., N-1
        # G(N) = X(N) - 1.0
        # The objective is sum of G(i)^2 (L2 group type)

        obj = 0.0

        # G(1) = X(1) - 1.0
        g1 = y[0] - 1.0
        obj += g1 * g1

        # G(i) = X(i+1) - X(i)^2 for i = 2, ..., N-1
        for i in range(1, self.n - 1):
            gi = y[i + 1] - y[i] * y[i]
            obj += gi * gi

        # G(N) = X(N) - 1.0
        gn = y[self.n - 1] - 1.0
        obj += gn * gn

        return jnp.array(obj)

    @property
    def bounds(self):
        # Default lower bound is 0.0 in CUTEst when not specified
        # All variables have upper bound 10.0 except X(N) which is free
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, 10.0)
        # X(N) is free (XR HATFLDC X(N))
        lower = lower.at[self.n - 1].set(-jnp.inf)
        upper = upper.at[self.n - 1].set(jnp.inf)
        return lower, upper

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # According to the SIF file comment (line 91),
        # the optimal objective value is 0.0
        return jnp.array(0.0)
