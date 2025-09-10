from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


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


class OSBORNE1(AbstractNonlinearEquations):
    """Osborne first problem in 5 variables (nonlinear equation version).

    This is a nonlinear equation version of problem OSBORNEA.
    This function is a nonlinear least squares with 33 groups. Each
    group has 2 nonlinear elements and one linear element.

    Source: Problem 17 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#32 (p. 77).

    SIF input: Ph. Toint, Dec 1989.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    classification NOR2-MN-5-33
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5  # Number of variables
    m: int = 33  # Number of groups

    def constraint(self, y: Any) -> tuple[Any, None]:
        """Returns the residuals as equality constraints."""
        x1, x2, x3, x4, x5 = y

        # Vectorized computation for all i from 0 to m-1
        i_vals = jnp.arange(self.m, dtype=jnp.float64)
        ti = -10.0 * i_vals

        # Element A: x2 * exp(x4 * ti) for all i
        element_a = x2 * jnp.exp(x4 * ti)

        # Element B: x3 * exp(x5 * ti) for all i
        element_b = x3 * jnp.exp(x5 * ti)

        # Residuals: x1 + element_a + element_b - y_data[i] for all i
        residuals = x1 + element_a + element_b - _Y_DATA

        return residuals, None

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
        # For nonlinear equations, objective is typically 0
        return jnp.array(0.0)

    @property
    def bounds(self) -> tuple[Any, Any] | None:
        """No bounds for this problem."""
        return None
