"""Fit a model containing a pole, given data - least squares version.

The problem is to fit a model containing a pole, given data
for values, first and second derivatives at two distinct points.
This is a least-squares version of problem PFIT2.

The problem is not convex.

SIF input: Ph. Toint, March 1994.
           Lower bound on H added, Nov 2002.

classification SBR2-AN-3-0
"""

import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class PFIT2LS(AbstractBoundedMinimisation):
    """Pole fitting least squares problem #2."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem data
    CF: float = -26.6666666666
    CG: float = -60.4444444444
    CH: float = -71.1111111111

    @property
    def n(self):
        """Number of variables."""
        return 3

    @property
    def y0(self):
        """Initial guess."""
        return jnp.array([1.0, 0.0, 1.0])  # A, R, H

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the objective function (sum of squared residuals)."""
        del args  # Not used

        A = y[0]
        R = y[1]
        H = y[2]

        # T1: A * R * H
        t1 = A * R * H

        # T2: A * R * H * (1 - (1+H)^(-A-1))
        y_val = 1.0 + H
        c = jnp.power(y_val, -(A + 1.0))
        b = 1.0 - c
        t2 = A * R * H * b

        # T3: A * (A + 1) * R * H^2
        t3 = A * (A + 1.0) * R * H * H

        # T4: R * (1 - (1+H)^(-A))
        c4 = jnp.power(y_val, -A)
        b4 = 1.0 - c4
        t4 = R * b4

        # T5: A * (A + 1) * R * H^2 * (1 - (1+H)^(-A-2))
        c5 = jnp.power(y_val, -(A + 2.0))
        b5 = 1.0 - c5
        t5 = A * (A + 1.0) * R * H * H * b5

        # Residuals from GROUP USES
        ef = -0.5 * t3 + t1 - t4 - self.CF
        eg = -t3 + t2 - self.CG
        eh = -t5 - self.CH

        # Return sum of squares (L2 group type)
        return ef * ef + eg * eg + eh * eh

    @property
    def bounds(self):
        """Bounds on variables."""
        # H has a lower bound of -0.5
        lower = jnp.array([-jnp.inf, -jnp.inf, -0.5])
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf])
        return (lower, upper)

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return jnp.array([1.0, 3.0, 2.0])

    @property
    def expected_objective_value(self):
        """Expected objective value is 0.0."""
        return jnp.array(0.0)
