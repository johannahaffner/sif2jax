import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO Requires human review, Claude tried and failed (and I don't know what was tried)
class BOOTH(AbstractConstrainedMinimisation):
    """Booth quadratic problem in 2 variables.

    Source: Problem 36 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: NLR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 2

    @property
    def m(self):
        """Number of constraints."""
        return 2

    def objective(self, y, args):
        """Compute the objective function.

        Looking at the SIF file, the objective is the sum of squared residuals:
        f(x) = (x1 + 2*x2 - 7)^2 + (2*x1 + x2 - 5)^2
        """
        del args
        x1, x2 = y

        # Group residuals
        g1 = x1 + 2.0 * x2 - 7.0
        g2 = 2.0 * x1 + x2 - 5.0

        # Sum of squares
        return g1**2 + g2**2

    def constraint(self, y):
        """Compute the constraints.

        From AMPL, the constraints are:
        x1 + 2*x2 - 7 = 0
        2*x1 + x2 - 5 = 0
        """
        x1, x2 = y

        eq_constraints = jnp.array([x1 + 2.0 * x2 - 7.0, 2.0 * x1 + x2 - 5.0])

        return eq_constraints, None

    def equality_constraints(self):
        """Both constraints are equalities."""
        return jnp.ones(2, dtype=bool)

    def y0(self):
        """Initial guess."""
        # No explicit start point in SIF, using zeros
        return jnp.zeros(2)

    def args(self):
        """No additional arguments."""
        return None

    def bounds(self):
        """All variables are free."""
        return None

    def expected_result(self):
        """Expected optimal solution.

        The linear system has unique solution:
        x1 + 2*x2 = 7
        2*x1 + x2 = 5

        Solving: x1 = 1, x2 = 3
        """
        return jnp.array([1.0, 3.0])

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # At the solution, both residuals are 0, so f = 0
        return jnp.array(0.0)
