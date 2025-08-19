"""DENSCHNB problem from CUTEst collection.

Classification: SUR2-AN-2-0

Source: an example problem (p. 201) in
J.E. Dennis and R.B. Schnabel,
"Numerical Methods for Unconstrained Optimization and Nonlinear Equations",
Prentice-Hall, Englewood Cliffs, 1983.

SIF input: Ph. Toint, Nov 1990.
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class DENSCHNB(AbstractUnconstrainedMinimisation):
    """DENSCHNB problem from CUTEst collection.

    Unconstrained minimization with 2 variables.
    Sum of squares problem.
    """

    n: int = 2
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Initial guess."""
        return jnp.ones(2)

    def objective(self, y, args):
        """Compute the objective function.

        The objective is a sum of squares:
        (x1 - 2)^2 + ((x1 - 2) * x2)^2 + (x2 + 1)^2
        """
        del args
        x1, x2 = y[0], y[1]

        # A group: (x1 - 2)^2
        term_a = (x1 - 2.0) ** 2

        # B group: ((x1 - 2) * x2)^2
        term_b = ((x1 - 2.0) * x2) ** 2

        # C group: (x2 + 1)^2
        term_c = (x2 + 1.0) ** 2

        return term_a + term_b + term_c

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Optimal point."""
        return jnp.array([2.0, -1.0])

    @property
    def expected_objective_value(self):
        """Optimal objective value."""
        return jnp.array(0.0)
