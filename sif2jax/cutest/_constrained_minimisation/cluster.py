import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CLUSTER(AbstractConstrainedMinimisation):
    """CLUSTER problem as a constrained formulation.

    Source: problem 207 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: NOR2-AN-2-2
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
        return 2  # Two residual equations

    def objective(self, y, args):
        """Compute the objective function.

        For a constrained least squares formulation, the objective is 0.
        """
        del args, y
        return jnp.array(0.0)

    def constraint(self, y):
        """Implement the abstract constraint method."""
        eq, ineq = self.equality_constraints(y, self.args())
        return eq, ineq

    def equality_constraints(self, y, args):
        """Compute the equality constraints.

        The residuals are:
        G1 = (X1 - X2^2) * (X1 - sin(X2))
        G2 = (cos(X2) - X1) * (X2 - cos(X1))
        """
        del args
        x1, x2 = y

        # Element A: G1
        f1_a = x1 - x2**2
        f2_a = x1 - jnp.sin(x2)
        g1 = f1_a * f2_a

        # Element B: G2
        f1_b = jnp.cos(x2) - x1
        f2_b = x2 - jnp.cos(x1)
        g2 = f1_b * f2_b

        return jnp.array([g1, g2]), None

    def y0(self):
        """Initial guess."""
        return jnp.zeros(2)

    def bounds(self):
        """Variable bounds."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    def args(self):
        """No additional arguments."""
        return None

    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
