"""Problem 14 (chained and modified HS53) in the paper.

L. Luksan
Hybrid methods in large sparse nonlinear least squares
J. Optimization Theory & Applications 89(3) 575-595 (1996)

SIF input: Nick Gould, June 2017.

Classification: NOR2-AN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class LUKSAN14(AbstractNonlinearEquations):
    """LUKSAN14 problem - chained and modified HS53."""

    _s: int = 32
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 3 * self._s + 2

    @property
    def m(self):
        """Number of residuals/equations."""
        return 7 * self._s

    @property
    def y0(self):
        """Initial guess."""
        return jnp.full(self.n, -1.0)

    @property
    def args(self):
        """No additional arguments."""
        return None

    def residual(self, x, args):
        """Compute the residual vector."""
        del args  # Not used

        s = self._s
        m = self.m

        res = jnp.zeros(m)

        # Loop over blocks
        i = 0  # 0-indexed (SIF uses 1-indexed)
        k = 0  # 0-indexed

        for j in range(s):
            # Element indices (converting from 1-indexed to 0-indexed)
            i1 = i + 1
            i2 = i + 2
            i3 = i + 3
            i4 = i + 4

            # Elements
            e_k = x[i] ** 2
            e_k6 = x[i1] ** 2

            # Equations E(K) through E(K+6)
            # E(K): -10*X(I+1) + 10*E(K) where E(K) = X(I)^2
            res = res.at[k].set(-10.0 * x[i1] + 10.0 * e_k)

            # E(K+1): X(I+1) + X(I+2) - 2.0
            res = res.at[k + 1].set(x[i1] + x[i2] - 2.0)

            # E(K+2): X(I+3) - 1.0
            res = res.at[k + 2].set(x[i3] - 1.0)

            # E(K+3): X(I+4) - 1.0
            res = res.at[k + 3].set(x[i4] - 1.0)

            # E(K+4): X(I) + 3*X(I+1)
            res = res.at[k + 4].set(x[i] + 3.0 * x[i1])

            # E(K+5): X(I+2) + X(I+3) - 2*X(I+4)
            res = res.at[k + 5].set(x[i2] + x[i3] - 2.0 * x[i4])

            # E(K+6): -10*X(I+4) + 10*E(K+6) where E(K+6) = X(I+1)^2
            res = res.at[k + 6].set(-10.0 * x[i4] + 10.0 * e_k6)

            # Update indices
            i += 3
            k += 7

        return res

    def constraint(self, y):
        """Return the constraint values as required by the abstract base class."""
        # For nonlinear equations, residuals are equality constraints
        return self.residual(y, self.args), None

    @property
    def bounds(self):
        """Bounds on variables."""
        # All variables are free
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value is 0.0 for nonlinear equations."""
        return jnp.array(0.0)
