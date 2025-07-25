"""Himmelblau 4 variable data fitting problem."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class HIMMELBF(AbstractUnconstrainedMinimisation):
    """Himmelblau 4 variable data fitting problem.

    A 4 variables data fitting problems by Himmelblau.

    Source: problem 32 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.

    See Buckley#76 (p. 66)

    SIF input: Ph. Toint, Dec 1989.

    classification SUR2-AN-4-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem data
    a_data: tuple[float, ...] = eqx.field(
        default=(0.0, 0.000428, 0.001000, 0.001610, 0.002090, 0.003480, 0.005250),
        init=False,
    )
    b_data: tuple[float, ...] = eqx.field(
        default=(7.391, 11.18, 16.44, 16.20, 22.20, 24.02, 31.32), init=False
    )

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function.

        The objective is a sum of squared ratios.
        """
        x1, x2, x3, x4 = y

        obj = 0.0
        for a, b in zip(self.a_data, self.b_data):
            # Element function HF
            u = x1 * x1 + a * x2 * x2 + a * a * x3 * x3
            v = b * (1.0 + a * x4 * x4)
            f_i = u / v

            # Group evaluation with constant 1.0 and scale 0.0001
            g_i = (f_i - 1.0) ** 2
            obj = obj + g_i / 0.0001  # SCALE means division

        return obj

    @property
    def y0(self):
        return jnp.array([2.7, 90.0, 1500.0, 10.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return jnp.array(318.572)
