import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class BOOTH(AbstractNonlinearEquations):
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

    def num_residuals(self):
        """Number of residual equations."""
        return 2

    def residual(self, y, args):
        """Compute the residuals.

        The nonlinear equations are:
        x1 + 2*x2 - 7 = 0
        2*x1 + x2 - 5 = 0
        """
        del args
        x1, x2 = y

        return jnp.array([x1 + 2.0 * x2 - 7.0, 2.0 * x1 + x2 - 5.0])

    def y0(self):
        """Initial guess."""
        # No explicit start point in SIF, using zeros
        return jnp.zeros(2)

    def args(self):
        """No additional arguments."""
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
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
