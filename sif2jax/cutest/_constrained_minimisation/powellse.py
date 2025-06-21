import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class POWELLSE(AbstractConstrainedMinimisation):
    """POWELLSE problem - The extended Powell singular problem.

    This problem is a sum of n/4 sets of four terms, each of which is
    assigned its own group. This is a nonlinear equation version
    of problem POWELLSG.

    Source: Problem 13 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Toint#19, Buckley#34 (p.85)

    SIF input: Ph. Toint, Dec 1989.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    Classification: NOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables (must be a multiple of 4)."""
        return 4  # Default value

    @property
    def m(self):
        """Number of constraints."""
        # Each set of 4 variables produces 4 equality constraints
        return self.n

    def objective(self, y, args):
        """Compute the objective (constant zero)."""
        del args, y
        # POWELLSE has a constant zero objective
        # The equations are handled as constraints
        return jnp.array(0.0)

    def constraint(self, y):
        """Compute the constraints."""
        x = y

        # Ensure n is a multiple of 4
        assert self.n % 4 == 0, "n must be a multiple of 4"

        # Equality constraints:
        eq_constraints = []

        for i in range(0, self.n, 4):
            # G(i): x(i) + 10*x(i+1) = 0
            g1 = x[i] + 10.0 * x[i + 1]

            # G(i+1): 5 * (x(i+2) - x(i+3)) = 0
            # Note: SIF has SCALE 0.2, which means multiply by 5 = 1/0.2
            g2 = 5.0 * (x[i + 2] - x[i + 3])

            # G(i+2): (x(i+1) - 2*x(i+2))^2 = 0
            g3 = (x[i + 1] - 2.0 * x[i + 2]) ** 2

            # G(i+3): 10 * (x(i) - x(i+3))^2 = 0
            # Note: SIF has SCALE 0.1, and squared term gives factor 10
            g4 = 10.0 * (x[i] - x[i + 3]) ** 2

            eq_constraints.extend([g1, g2, g3, g4])

        eq_constraints = jnp.array(eq_constraints)
        # No inequality constraints
        return eq_constraints, None

    def y0(self):
        """Initial guess."""
        x0 = jnp.zeros(self.n)
        for i in range(0, self.n, 4):
            x0 = x0.at[i].set(3.0)
            x0 = x0.at[i + 1].set(-1.0)
            x0 = x0.at[i + 2].set(0.0)
            x0 = x0.at[i + 3].set(1.0)
        return x0

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
