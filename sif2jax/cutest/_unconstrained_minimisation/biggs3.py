"""Biggs EXP problem in 3 variables."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class BIGGS3(AbstractUnconstrainedMinimisation):
    """Biggs EXP problem in 3 variables.

    This function is a nonlinear least squares with 13 groups. Each
    group has 3 nonlinear elements. It is obtained by fixing
    X3 = 1, X5 = 4 and X6 = 3 in BIGGS6.

    Source: Problem 152 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    classification SXR2-AN-6-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 3  # Number of free variables
    m: int = 13  # Number of groups

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function."""
        x1, x2, x4 = y
        # Fixed values
        x3 = 1.0
        x5 = 4.0
        x6 = 3.0

        obj = 0.0

        for i in range(1, self.m + 1):
            ti = -0.1 * i

            # Element A: V1 * exp(T * V2) with V1=X3, V2=X1, T=ti
            a_i = x3 * jnp.exp(ti * x1)

            # Element B: V1 * exp(T * V2) with V1=X4, V2=X2, T=ti
            b_i = x4 * jnp.exp(ti * x2)

            # Element C: V1 * exp(T * V2) with V1=X6, V2=X5, T=ti
            c_i = x6 * jnp.exp(ti * x5)

            # Constants y_i from SIF
            emti = jnp.exp(ti)
            e2 = jnp.exp(-i)
            e3 = jnp.exp(4.0 * ti)
            y_i = emti - 5.0 * e2 + 3.0 * e3

            # Group: L2 type (squared difference)
            # G(i) = (a_i - b_i + c_i - y_i)^2
            g_i = a_i - b_i + c_i - y_i
            obj = obj + g_i * g_i

        return obj

    @property
    def y0(self):
        # Only free variables: x1, x2, x4
        return jnp.array([1.0, 2.0, 1.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return jnp.array(0.0)
