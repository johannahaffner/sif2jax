import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class POWELLSQ(AbstractConstrainedMinimisation):
    """POWELLSQ problem - Powell's singular problem.

    Source:
    M.J.D. Powell,
    "A hybrid method for nonlinear equations",
    In P. Rabinowitz(ed.) "Numerical Methods for Nonlinear Algebraic
    Equations", Gordon and Breach, 1970.

    See also Buckley#217 (p.84.)

    Classification: NOR2-AN-2-2

    SIF input: Ph. Toint, Dec 1989, correction November 2002.
               NIMG corrected July 2005 (thanks to Roger Fletcher)
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
        # 2 equality constraints
        return 2

    def objective(self, y, args):
        """Compute the objective (constant zero)."""
        del args, y
        # POWELLSQ has a constant zero objective
        # The equations are handled as constraints
        return jnp.array(0.0)

    def constraint(self, y):
        """Compute the constraints."""
        x1, x2 = y

        # Equality constraints:
        # Group G1: x1^2 = 0
        g1 = x1**2

        # Group G2: 10*x1/(x1+0.1) + 2*x2^2 = 0
        g2 = 10.0 * x1 / (x1 + 0.1) + 2.0 * x2**2

        eq_constraints = jnp.array([g1, g2])
        # No inequality constraints
        return eq_constraints, None

    def y0(self):
        """Initial guess."""
        return jnp.array([3.0, 1.0])

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def bounds(self):
        """Variable bounds (all free)."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
