import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractBoundedQuadraticProblem


class BQP1VAR(AbstractBoundedQuadraticProblem):
    """BQP1VAR problem - a one variable box-constrained quadratic.

    Source: a one variable box-constrained quadratic

    SIF input: Nick Gould, March 1992

    classification QBR2-AN-1-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 1

    def objective(self, y, args):
        """Compute the objective.

        The objective has two groups:
        G1: x1 (linear term)
        G2: x1 with L2 group type, which applies GVAR^2 where GVAR = x1

        So the objective is: x1 + x1^2
        """
        del args
        x1 = y[0]

        # G1: linear term (coefficient 1.0)
        g1 = x1

        # G2: quadratic term (L2 group type applies GVAR^2)
        # F = GVAR * GVAR where GVAR = x1
        g2 = x1 * x1

        return g1 + g2

    @property
    def y0(self):
        """Initial guess."""
        return inexact_asarray(jnp.array([0.25]))

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Variable bounds."""
        # 0.0 <= x1 <= 0.5
        lower = jnp.array([0.0])
        upper = jnp.array([0.5])
        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # The minimum of x + x^2 on [0, 0.5] occurs at x = 0
        # (derivative is 1 + 2x, which is positive on [0, 0.5])
        return jnp.array([0.0])

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file comment: Solution = 0.0
        return jnp.array(0.0)
