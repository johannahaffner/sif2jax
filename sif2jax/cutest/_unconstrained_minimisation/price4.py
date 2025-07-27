from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class PRICE4(AbstractUnconstrainedMinimisation):
    """SCIPY global optimization benchmark example PRICE4.

    Fit: (2x_1^2 x_2 - x_2^3, 6x_1 - x_2^2 + x_2) + e = 0

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/
            benchmarks/go_benchmark_functions

    SIF input: Nick Gould, Jan 2020

    classification SUR2-MN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Number of variables
    m: int = 2  # Number of groups

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function."""
        x1, x2 = y

        # Element E12 (CUBEL type): x2 * x1^3
        e12 = x2 * x1**3

        # Element E1 (CUBE type): x2^3
        e1 = x2**3

        # Element E2 (SQR type): x2^2
        e2 = x2**2

        # Group F1: 2.0 * e12 - 1.0 * e1
        f1 = 2.0 * e12 - e1

        # Group F2: 6.0 * x1 + x2 - e2
        f2 = 6.0 * x1 + x2 - e2

        # L2 group type: sum of squares
        obj = f1**2 + f2**2

        return obj

    @property
    def y0(self):
        return jnp.array([1.0, 5.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Not provided explicitly in SIF file
        return None

    @property
    def expected_objective_value(self):
        # From SIF file comment: 0.0
        return jnp.array(0.0)
