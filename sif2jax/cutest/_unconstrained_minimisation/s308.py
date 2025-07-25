"""Schittkowski problem 308."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class S308(AbstractUnconstrainedMinimisation):
    """Schittkowski problem 308.

    A 2-variable unconstrained minimization problem with quadratic
    and trigonometric terms.

    Source: problem 308 in
    K. Schittkowski,
    "More Test Problems for Nonlinear Programming Codes",
    Springer Verlag, Berlin, 1987.

    SIF input: Ph. Toint, April 1991.

    classification SUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function."""
        x1, x2 = y

        # Element functions
        # QUAD: x1^2 + x1*x2 + x2^2
        quad = x1 * x1 + x1 * x2 + x2 * x2

        # SINE: sin(x1)
        sine = jnp.sin(x1)

        # COSN: cos(x2)
        cosn = jnp.cos(x2)

        # Group evaluations with L2 group type (squares each term)
        obj = quad * quad + sine * sine + cosn * cosn

        return obj

    @property
    def y0(self):
        return jnp.array([3.0, 0.1])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        # From SIF file
        return jnp.array(0.773199)
