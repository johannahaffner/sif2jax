import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class RECIPE(AbstractNonlinearEquations):
    """RECIPE problem - Nonlinear equation system.

    Source: problem 155 (p. 88) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: NOR2-AY-3-3

    This is a system of 3 nonlinear equations in 3 variables:
    G1: X1 - 5 = 0
    G2: X2² = 0
    G3: X3/(X2-X1) = 0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(3)

    def constraint(self, y):
        """Compute the nonlinear equations."""
        x1, x2, x3 = y[0], y[1], y[2]

        # G1: X1 - 5 = 0
        g1 = x1 - 5.0

        # G2: X2² = 0
        g2 = x2 * x2

        # G3: X3/(X2-X1) = 0
        # Using the XOVERU element: V/U where V=X3, U=X2-X1
        u = x2 - x1
        g3 = x3 / u

        equality_constraints = jnp.array([g1, g2, g3])
        inequality_constraints = None

        return equality_constraints, inequality_constraints

    @property
    def y0(self):
        """Initial guess."""
        return jnp.array([2.0, 5.0, 1.0])

    @property
    def args(self):
        """Additional arguments."""
        return None

    @property
    def bounds(self):
        """Bounds on variables."""
        # All variables are free (FR RECIPE 'DEFAULT')
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # From the constraint structure:
        # G1: X1 = 5
        # G2: X2 = 0
        # G3: X3/(X2-X1) = 0 → X3/(0-5) = 0 → X3 = 0
        return jnp.array([5.0, 0.0, 0.0])

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # For nonlinear equations, objective is typically sum of squares of constraints
        return jnp.array(0.0)
