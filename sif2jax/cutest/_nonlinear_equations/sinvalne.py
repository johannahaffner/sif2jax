from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class SINVALNE(AbstractNonlinearEquations):
    """A trigonometric variant of the 2 variables Rosenbrock "banana valley" problem.

    This problem is a nonlinear equation version of problem SINEVAL.

    Source: problem 4.2 in
    Y. Xiao and F. Zhou,
    "Non-monotone trust region methods with curvilinear path
    in unconstrained optimization",
    Computing, vol. 48, pp. 303-317, 1992.

    SIF input: F Facchinei, M. Roma and Ph. Toint, June 1994

    classification NOR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Number of variables
    m: int = 2  # Number of equations

    def residual(self, y: Any, args: Any) -> Any:
        """Compute the residuals."""
        x1, x2 = y

        # Group G1: c * (x2 - sin(x1)) where c = 0.1
        # Note: pycutest inverts SCALE parameters for NLE problems
        # SCALE C=0.1 becomes 1/0.1 = 10.0
        g1 = 10.0 * (x2 - jnp.sin(x1))

        # Group G2: 2 * x1
        # Note: SCALE 2.0 becomes 1/2.0 = 0.5
        g2 = 0.5 * x1

        return jnp.array([g1, g2])

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def y0(self):
        return jnp.array([4.712389, -1.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # At the solution, both residuals should be zero
        # This gives: x2 = sin(x1) and x1 = 0
        # So x1 = 0 and x2 = sin(0) = 0
        return jnp.array([0.0, 0.0])

    @property
    def expected_objective_value(self):
        # For nonlinear equations, objective is sum of squares of residuals
        return jnp.array(0.0)

    @property
    def bounds(self) -> None:
        """No bounds for this problem."""
        return None
