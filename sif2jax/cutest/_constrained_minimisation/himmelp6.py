from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HIMMELP6(AbstractConstrainedMinimisation):
    """A nonlinear problem with inequality constraints, attributed to Himmelblau
    by B.N. Pshenichnyj (case V).

    The problem is nonconvex and has redundant constraints at the solution.

    Source:
    B.N. Pshenichnyj
    "The Linearization Method for Constrained Optimization",
    Springer Verlag, SCM Series 22, Heidelberg, 1994

    SIF input: Ph. Toint, December 1994.

    classification OQR2-AN-2-5
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Number of variables

    # Problem data
    b1 = 0.1963666677 + 75.0  # 75.1963666677
    b2 = -0.8112755343 - 3.0  # -3.8112755343
    b6 = -0.8306567613 - 6.0  # -6.8306567613

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function."""
        x1, x2 = y

        # Constants from the SIF file
        b3 = 0.1269366345
        b4 = -0.20567665 * 0.01
        b5 = 0.103450e-4
        b7 = 0.0302344793
        b8 = -0.12813448 * 0.01
        b9 = 0.352599e-4
        b10 = -0.2266e-6
        b11 = 0.2564581253
        b12 = -0.003460403
        b13 = 0.135139e-4
        b14 = -0.1064434908 - 28.0
        b15 = -0.52375e-5
        b16 = -0.63e-8
        b17 = 0.7e-9
        b18 = 0.3405462 * 0.001
        b19 = -0.16638e-5
        b20 = -2.86731123 - 0.92e-8

        # Element OB (OBNL type) - same as HIMMELP2/P3/P4/P5
        A = b7 * x1 + b8 * x1**2 + b9 * x1**3 + b10 * x1**4
        B = b18 * x1 + b15 * x1**2 + b16 * x1**3
        C = b3 * x1**2 + b4 * x1**3 + b5 * x1**4
        F = b11 * x2**2 + b12 * x2**3 + b13 * x2**4
        G = b17 * x1**3 + b19 * x1
        E = jnp.exp(0.0005 * x1 * x2)

        element_ob = C + x2 * A + F + b14 / (1.0 + x2) + B * x2**2 + G * x2**3 + b20 * E

        # Objective: b1 + b2 * x1 + b6 * x2 - element_ob
        obj = self.b1 + self.b2 * x1 + self.b6 * x2 - element_ob

        return obj

    def constraint(self, y: Any):
        """Compute the constraints.

        C5: x1 * x2 >= 700.0
        C8: -x2 + 0.008 * x1^2 >= 0
        C9: 5 * x1 + 100 * x2 - x2^2 >= 2775
        C11: x1 - 1.5 * x2 >= 22.5
        C12: -x1 + 0.16 * x2 >= 41.4
        """
        x1, x2 = y

        # Element X1X2 (2PR type): x1 * x2
        element_x1x2 = x1 * x2

        # Element X1SQ (SQ type): x1^2
        element_x1sq = x1 * x1

        # Element X2SQ (SQ type): x2^2
        element_x2sq = x2 * x2

        # Constraint C5: element_x1x2 - 700 >= 0
        c5 = element_x1x2 - 700.0

        # Constraint C8: -x2 + 0.008 * element_x1sq >= 0
        c8 = -x2 + 0.008 * element_x1sq

        # Constraint C9: 5 * x1 + 100 * x2 - element_x2sq - 2775 >= 0
        c9 = 5.0 * x1 + 100.0 * x2 - element_x2sq - 2775.0

        # Constraint C11: x1 - 1.5 * x2 - 22.5 >= 0
        c11 = x1 - 1.5 * x2 - 22.5

        # Constraint C12: -x1 + 0.16 * x2 - 41.4 >= 0
        c12 = -x1 + 0.16 * x2 - 41.4

        return None, jnp.array([c5, c8, c9, c11, c12])

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return (jnp.array([0.0, 0.0]), jnp.array([75.0, 65.0]))

    @property
    def y0(self):
        return jnp.array([68.8, 31.2])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # From SIF file comment
        return jnp.array([75.0, 65.0])

    @property
    def expected_objective_value(self):
        # From SIF file comment: -59.01312394 (same as HIMMELP3/P4/P5)
        return jnp.array(-59.01312394)
