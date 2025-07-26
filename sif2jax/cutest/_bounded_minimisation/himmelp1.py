"""Himmelblau problem 1 - nonlinear bounded optimization.

# TODO: Human review needed
# Attempts made:
# 1. Fixed B1 calculation from multiplication to addition
# 2. Corrected group coefficient interpretation (-B2, -B6)
# 3. Verified element function calculation
# Suspected issues: SIF format interpretation, objective function structure
# Additional resources needed: OBNL element type clarification
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HIMMELP1(AbstractBoundedMinimisation):
    """Himmelblau problem 1.

    A nonlinear problem with inequality constraints, attributed to Himmelblau
    by B.N. Pshenichnyj (case 0: only bounds).

    Source:
    B.N. Pshenichnyj
    "The Linearization Method for Constrained Optimization",
    Springer Verlag, SCM Series 22, Heidelberg, 1994

    SIF input: Ph. Toint, December 1994.

    classification OBR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function."""
        x1, x2 = y

        # Constants from SIF file
        # B1 = 0.1963666677 + 75.0 = 75.1963666677
        # B2 = -0.8112755343 + (-3.0) = -3.8112755343
        # B6 = -0.8306567613 + (-6.0) = -6.8306567613
        # But group coefficients are -B2 and -B6, so we store the positive values
        b1 = 0.1963666677 + 75.0  # 75.1963666677
        b2 = -(-0.8112755343 - 3.0)  # -(-3.8112755343) = 3.8112755343
        b6 = -(-0.8306567613 - 6.0)  # -(-6.8306567613) = 6.8306567613

        # Additional constants used in the element function
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

        # Compute element function OBNL
        a = b7 * x1 + b8 * x1**2 + b9 * x1**3 + b10 * x1**4
        b = b18 * x1 + b15 * x1**2 + b16 * x1**3
        c = b3 * x1**2 + b4 * x1**3 + b5 * x1**4
        f = b11 * x2**2 + b12 * x2**3 + b13 * x2**4
        g = b17 * x1**3 + b19 * x1
        e = jnp.exp(0.0005 * x1 * x2)

        element_value = (
            c + x2 * a + f + b14 / (1.0 + x2) + b * x2**2 + g * x2**3 + b20 * e
        )

        # Objective function: B1 + (-B2)*X1 + (-B6)*X2 + (-1.0)*element_ob
        # Since b2 and b6 already store the negated values, use them directly
        obj = b1 + b2 * x1 + b6 * x2 - element_value

        return obj

    @property
    def bounds(self):
        """Variable bounds: 0 <= x1 <= 95, 0 <= x2 <= 75."""
        lower = jnp.array([0.0, 0.0])
        upper = jnp.array([95.0, 75.0])
        return lower, upper

    @property
    def y0(self):
        return jnp.array([95.0, 10.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution from SIF file
        return jnp.array([81.192, 69.158])

    @property
    def expected_objective_value(self):
        # From SIF file
        return jnp.array(-62.053869846)
