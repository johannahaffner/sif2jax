from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class MEXHAT(AbstractUnconstrainedMinimisation):
    """The mexican hat problem with penalty parameter 0.00001.

    Source:
    A.A. Brown and M. Bartholomew-Biggs,
    "Some effective methods for unconstrained optimization based on
    the solution of ordinary differential equations",
    Technical Report 178, Numerical Optimization Centre, Hatfield
    Polytechnic, (Hatfield, UK), 1987.

    SIF input: Ph. Toint, June 1990.

    classification OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Number of variables

    # Penalty parameter
    invp = 0.00001

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function."""
        x1, x2 = y

        # Element O1 (SSQ type with S=1.0)
        xms = x1 - 1.0
        o1 = xms * xms

        # Element O2 (SSQ type with S=1.0)
        # Both O1 and O2 use X1 and S=1.0, so they are identical
        o2 = o1

        # Element C1 (XMYSQ type)
        xx = x2 - x1 * x1
        c1 = xx * xx

        # Group F: E  F         O1        -1.0           O2        -1.0
        f = -1.0 * o1 + -1.0 * o2  # = -2.0 * o1 since o1 == o2

        # Group C with L2 type and constant 0.02
        c_linear = 10000.0 * c1 + o1 - 0.02
        c_group = c_linear * c_linear

        # Objective with scale - group C scaled by INVP means divided by INVP
        obj = f + c_group / self.invp

        return obj

    @property
    def y0(self):
        return jnp.array([0.86, 0.72])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        """Expected solution - not provided in SIF file."""
        # Will be determined by testing
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file comments, two possible solutions:
        # -0.0898793 or -1.1171526
        # Will use the first one
        return jnp.array(-0.0898793)
