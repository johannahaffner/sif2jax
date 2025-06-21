import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class POWELLBS(AbstractConstrainedMinimisation):
    """POWELLBS problem - Powell badly scaled problem.

    This problem is a sum of n-1 sets of 2 groups, both involving
    nonlinear elements and being of the least square type.
    Its Hessian matrix is tridiagonal.

    Source: Problem 3 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Toint#34, Buckley#22 (p. 82).

    SIF input: Ph. Toint, Dec 1989.

    Classification: NOR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 2  # Default size

    @property
    def m(self):
        """Number of constraints."""
        # 2*(n-1) equality constraints
        return 2 * (self.n - 1)

    def objective(self, y, args):
        """Compute the objective (constant zero)."""
        del args, y
        # POWELLBS has a constant zero objective
        # The equations are handled as constraints
        return jnp.array(0.0)

    def constraint(self, y):
        """Compute the constraints."""
        x = y

        # Equality constraints:
        # For i = 1 to n-1:
        # A(i) = 10000 * x(i) * x(i+1) - 1 = 0
        # B(i) = exp(-x(i)) + exp(-x(i+1)) - 1.0001 = 0
        eq_constraints = []
        for i in range(self.n - 1):
            a_i = 10000.0 * x[i] * x[i + 1] - 1.0
            b_i = jnp.exp(-x[i]) + jnp.exp(-x[i + 1]) - 1.0001
            eq_constraints.append(a_i)
            eq_constraints.append(b_i)

        eq_constraints = jnp.array(eq_constraints)
        # No inequality constraints
        return eq_constraints, None

    def y0(self):
        """Initial guess."""
        x0 = jnp.zeros(self.n)
        x0 = x0.at[0].set(0.0)
        x0 = x0.at[1].set(1.0)
        return x0

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds (all free)."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
