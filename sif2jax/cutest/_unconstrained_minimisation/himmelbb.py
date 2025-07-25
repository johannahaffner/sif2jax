from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class HIMMELBB(AbstractUnconstrainedMinimisation):
    """A 2 variables problem by Himmelblau.

    Source: problem 27 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.

    See Buckley#77 (p. 62)

    SIF input: Ph. Toint, Dec 1989.

    classification OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Number of variables

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function."""
        x1, x2 = y

        # Element H
        r1 = x1 * x2
        r2 = 1.0 - x1
        r3 = 1.0 - x2 - x1 * r2**5

        element_h = r1 * r2 * r3

        # Group G with L2 type
        obj = element_h * element_h

        return obj

    @property
    def y0(self):
        return jnp.array([-1.2, 1.0])

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
        # From the SIF file: SOLTN 0.0
        return jnp.array(0.0)
