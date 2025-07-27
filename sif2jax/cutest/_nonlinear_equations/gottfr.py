from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class GOTTFR(AbstractNonlinearEquations):
    """The GOTTFR problem from Sisser.

    Source: problem 208 (p. 56) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    classification NQR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Number of variables
    m: int = 2  # Number of equations

    def residual(self, y: Any, args: Any) -> Any:
        """Compute the residual functions."""
        x1, x2 = y

        # Element E1 (ET1 type)
        fa = x1 + 3.0 * x2
        fb = 1.0 - x1
        e1 = -0.1136 * fa * fb

        # Element E2 (ET2 type)
        fa2 = 2.0 * x1 - x2
        fb2 = 1.0 - x2
        e2 = 7.5 * fa2 * fb2

        # Groups
        g1 = x1 + e1
        g2 = x2 + e2

        return jnp.array([g1, g2])

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        """No bounds for this problem."""
        return None

    @property
    def y0(self):
        return jnp.array([0.5, 0.5])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        """Expected solution - not provided in SIF file."""
        # Will be determined by testing
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
