"""Price problem 3."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class PRICE3(AbstractUnconstrainedMinimisation):
    """SCIPY global optimization benchmark example Price03.

    Fit: y = (10(x_1^2-x_2), sqrt(6)(6.4(x_2-0.5)^2 -x_1)) + e

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/
    benchmarks/go_benchmark_functions

    SIF input: Nick Gould, Jan 2020

    classification SUR2-MN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function."""
        x1, x2 = y

        # Element functions
        # E1: SQR type with x1: x1^2
        e1 = x1 * x1

        # E2: SSQR type with x2: (x2 - 0.5)^2
        e2 = (x2 - 0.5) ** 2

        # Group functions with L2 type (squares the result)
        # F1: (e1 - x2)^2 / 0.01 = (x1^2 - x2)^2 / 0.01 (SCALE means division)
        f1 = (e1 - x2) ** 2 / 0.01

        # F2: (6.4 * e2 - x1 - 0.6)^2
        # Testing against pycutest shows the constant should be subtracted, not added
        # This gives the correct objective value
        f2 = (6.4 * e2 - x1 - 0.6) ** 2

        return f1 + f2

    @property
    def y0(self):
        return jnp.array([1.0, 5.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The optimal solution should have objective value 0
        # This occurs when both terms are zero:
        # x1^2 = x2 and 6.4*(x2 - 0.5)^2 = x1 - 0.6
        return None

    @property
    def expected_objective_value(self):
        return jnp.array(0.0)
