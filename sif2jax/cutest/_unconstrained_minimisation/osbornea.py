from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# Data values from SIF file
_Y_DATA = jnp.array(
    [
        0.844,
        0.908,
        0.932,
        0.936,
        0.925,
        0.908,
        0.881,
        0.850,
        0.818,
        0.784,
        0.751,
        0.718,
        0.685,
        0.658,
        0.628,
        0.603,
        0.580,
        0.558,
        0.538,
        0.522,
        0.506,
        0.490,
        0.478,
        0.467,
        0.457,
        0.448,
        0.438,
        0.431,
        0.424,
        0.420,
        0.414,
        0.411,
        0.406,
    ]
)

# Pre-computed time values: ti = -10.0 * i for i from 0 to 32
_TI = jnp.array(
    [
        0.0,
        -10.0,
        -20.0,
        -30.0,
        -40.0,
        -50.0,
        -60.0,
        -70.0,
        -80.0,
        -90.0,
        -100.0,
        -110.0,
        -120.0,
        -130.0,
        -140.0,
        -150.0,
        -160.0,
        -170.0,
        -180.0,
        -190.0,
        -200.0,
        -210.0,
        -220.0,
        -230.0,
        -240.0,
        -250.0,
        -260.0,
        -270.0,
        -280.0,
        -290.0,
        -300.0,
        -310.0,
        -320.0,
    ]
)


class OSBORNEA(AbstractUnconstrainedMinimisation):
    """Osborne first problem in 5 variables.

    This function is a nonlinear least squares with 33 groups. Each
    group has 2 nonlinear elements and one linear element.

    Source: Problem 17 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#32 (p. 77).

    SIF input: Ph. Toint, Dec 1989.

    classification SUR2-MN-5-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5  # Number of variables
    m: int = 33  # Number of groups

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function (vectorized)."""
        x1, x2, x3, x4, x5 = y

        # Element A: x2 * exp(x4 * ti) for all i
        element_a = x2 * jnp.exp(x4 * _TI)

        # Element B: x3 * exp(x5 * ti) for all i
        element_b = x3 * jnp.exp(x5 * _TI)

        # Groups: (x1 + element_a + element_b - y_data[i])^2 for all i
        residuals = x1 + element_a + element_b - _Y_DATA

        # Sum of squared residuals
        return jnp.sum(residuals**2)

    @property
    def y0(self):
        return jnp.array([0.5, 1.5, -1.0, 0.01, 0.02])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Not provided explicitly in SIF file
        return None

    @property
    def expected_objective_value(self):
        # From SIF file comment: 5.46489D-05
        return jnp.array(5.46489e-05)
