from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class OSBORNEB(AbstractUnconstrainedMinimisation):
    """Osborne second problem in 11 variables.

    This function is a nonlinear least squares with 65 groups. Each
    group has 4 nonlinear elements.

    Source: Problem 19 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#32 (p.78).

    SIF input: Ph. Toint, Dec 1989.

    classification SUR2-MN-11-0

    TODO: Human review needed
    Attempts made: Fixed element A formula, verified element definitions
    Suspected issues: Possible issue with data mapping or element interpretation
    Additional resources needed: Comparison with working OSBORNE implementation
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 11  # Number of variables
    m: int = 65  # Number of groups

    # Data values
    y_data = jnp.array(
        [
            1.366,
            1.191,
            1.112,
            1.013,
            0.991,
            0.885,
            0.831,
            0.847,
            0.786,
            0.725,
            0.746,
            0.679,
            0.608,
            0.655,
            0.616,
            0.606,
            0.602,
            0.626,
            0.651,
            0.724,
            0.649,
            0.649,
            0.694,
            0.644,
            0.624,
            0.661,
            0.612,
            0.558,
            0.533,
            0.495,
            0.500,
            0.423,
            0.395,
            0.375,
            0.372,
            0.391,
            0.396,
            0.405,
            0.428,
            0.429,
            0.523,
            0.562,
            0.607,
            0.653,
            0.672,
            0.708,
            0.633,
            0.668,
            0.645,
            0.632,
            0.591,
            0.559,
            0.597,
            0.625,
            0.739,
            0.710,
            0.729,
            0.720,
            0.636,
            0.581,
            0.428,
            0.292,
            0.162,
            0.098,
            0.054,
        ]
    )

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function (vectorized)."""
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = y

        # Vectorized computation for all i from 1 to m
        # SIF uses IA I-1 I 1, meaning I-1 = I + 1
        # So for I from 1 to 65: I-1 = 2 to 66, and ti = 0.1 * (I-1) = 0.2 to 6.6
        # Using 0-based indexing: for array index 0 to 64, ti = 0.2 to 6.6
        i_minus_1_vals = jnp.arange(2, self.m + 2, dtype=jnp.float64)
        ti = 0.1 * i_minus_1_vals

        # Element A: x1 * exp(-ti * x5) (PEXP type)
        element_a = x1 * jnp.exp(-ti * x5)

        # Element B: x2 * exp(-(ti - x9)^2 * x6) (PEXP3 type)
        diff_b = ti - x9
        element_b = x2 * jnp.exp(-(diff_b * diff_b) * x6)

        # Element C: x3 * exp(-(ti - x10)^2 * x7) (PEXP3 type)
        diff_c = ti - x10
        element_c = x3 * jnp.exp(-(diff_c * diff_c) * x7)

        # Element D: x4 * exp(-(ti - x11)^2 * x8) (PEXP3 type)
        diff_d = ti - x11
        element_d = x4 * jnp.exp(-(diff_d * diff_d) * x8)

        # Residuals: element_a + element_b + element_c + element_d - y_data[i]
        residuals = element_a + element_b + element_c + element_d - self.y_data

        # Sum of squared residuals
        return jnp.sum(residuals * residuals)

    @property
    def y0(self):
        return jnp.array([1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Not provided explicitly in SIF file
        return None

    @property
    def expected_objective_value(self):
        # From SIF file comment: 0.04013774
        return jnp.array(0.04013774)
