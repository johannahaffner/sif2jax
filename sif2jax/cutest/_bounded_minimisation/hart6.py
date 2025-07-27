"""Hartman 6-dimensional test problem."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HART6(AbstractBoundedMinimisation):
    """Hartman 6-dimensional test problem.

    Source: Hartman problem 6 in
    L. C. W. Dixon and G. P. Szego (Eds.)
    Towards Global Optimization
    North Holland, 1975.
    Paper 9, page 163.

    SIF input: A.R. Conn May 1995

    classification OBR2-AN-6-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function."""
        # Coefficients c_i
        c = jnp.array([1.0, 1.2, 3.0, 3.2])

        # Matrix A (4x6)
        a = jnp.array(
            [
                [10.0, 0.05, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ]
        )

        # Matrix P (4x6)
        p = jnp.array(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )

        # Compute objective: sum of negative exponentials
        obj = 0.0
        for i in range(4):
            # Compute sum of a[i,j] * (x[j] - p[i,j])^2
            diff = y - p[i, :]
            weighted_sum = jnp.sum(a[i, :] * diff * diff)
            obj = obj - c[i] * jnp.exp(-weighted_sum)

        return obj

    @property
    def bounds(self):
        """Variable bounds: 0 <= x_i <= 1 for all i."""
        lower = jnp.zeros(6)
        upper = jnp.ones(6)
        return lower, upper

    @property
    def y0(self):
        return jnp.full(6, 0.2)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The optimal solution from the commented section in SIF
        return jnp.array([0.201661, 0.149985, 0.476919, 0.275317, 0.311688, 0.657283])

    @property
    def expected_objective_value(self):
        # From SIF file
        return jnp.array(-3.32288689158)
