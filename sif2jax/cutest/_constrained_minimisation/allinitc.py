"""ALLINITC problem.

A problem with "all in it". Intended to verify that changes to LANCELOT are safe.

Source:
N. Gould: private communication.

SIF input: Nick Gould, June 1990.

classification OOR2-AY-4-1
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class ALLINITC(AbstractConstrainedMinimisation):
    """ALLINITC problem implementation."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # Group FT1 (TRIVIAL)
        ft1 = 0.0

        # Group FT2 (TRIVIAL) with linear term
        ft2 = x3 - 1.0

        # Group FT3 (TRIVIAL) with element FT3E1 (SQR)
        ft3 = x1 * x1

        # Group FT4 (TRIVIAL) with elements FT4E1 (SQR) and FT4E2 (SQR2)
        ft4 = x2 * x2 + (x3 + x4) * (x3 + x4)

        # Group FT5 (TRIVIAL) with elements FT56E1 (SINSQR) and FT5E2 (PRODSQR)
        sinx3 = jnp.sin(x3)
        ft5 = (x4 - 3.0) + sinx3 * sinx3 + (x1 * x2) ** 2

        # Group FT6 (TRIVIAL) with element FT56E1 (SINSQR)
        ft6 = sinx3 * sinx3

        # Group FNT1 (L2)
        fnt1 = 0.0
        fnt1_l2 = fnt1 * fnt1

        # Group FNT2 (L2) with linear term
        fnt2 = x4 - 1.0
        fnt2_l2 = fnt2 * fnt2

        # Group FNT3 (L2) with element FNT3E1 (SQR)
        fnt3 = x2 * x2
        fnt3_l2 = fnt3 * fnt3

        # Group FNT4 (L2) with elements FNT4E1 (SQR) and FNT4E2 (SQR2)
        fnt4 = x3 * x3 + (x4 + x1) * (x4 + x1)
        fnt4_l2 = fnt4 * fnt4

        # Group FNT5 (L2) with elements FNT56E1 (SINSQR) and FNT5E2 (PRODSQR)
        sinx4 = jnp.sin(x4)
        fnt5 = (x1 - 4.0) + sinx4 * sinx4 + (x2 * x3) ** 2
        fnt5_l2 = fnt5 * fnt5

        # Group FNT6 (L2) with element FNT56E1 (SINSQR)
        fnt6 = sinx4 * sinx4
        fnt6_l2 = fnt6 * fnt6

        # Sum all groups with L2 norm for FNT groups
        obj = (
            ft1
            + ft2
            + ft3
            + ft4
            + ft5
            + ft6
            + fnt1_l2
            + fnt2_l2
            + fnt3_l2
            + fnt4_l2
            + fnt5_l2
            + fnt6_l2
        )

        return obj

    def constraint(self, y):
        """Compute the constraints."""
        x1, x2 = y[0], y[1]

        # E  C1        FT3E1                    FT4E1
        # Group C1 (TRIVIAL) with elements FT3E1 (SQR) and FT4E1 (SQR)
        # Equality constraint: x1^2 + x2^2 - 1.0 = 0
        c1 = x1 * x1 + x2 * x2 - 1.0

        # Return (equality_constraints, inequality_constraints)
        # This problem has 1 equality constraint and no inequality constraints
        return jnp.array([c1]), None

    @property
    def bounds(self):
        """Return the bounds on variables."""
        # Variable bounds:
        # FR ALLINITC  X1
        # LO ALLINITC  X2        1.0
        # LO ALLINITC  X3        -1.0D+20
        # UP ALLINITC  X3        1.0
        # FX ALLINITC  X4        2.0

        y_lwr = jnp.array([-jnp.inf, 1.0, -1.0e20, 2.0])
        y_upr = jnp.array([jnp.inf, jnp.inf, 1.0, 2.0])

        return y_lwr, y_upr

    @property
    def y0(self):
        """Return the initial point."""
        return jnp.zeros(4)

    @property
    def n(self):
        """Number of variables."""
        return 4

    @property
    def m(self):
        """Number of constraints."""
        return 1

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return None
