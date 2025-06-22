import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed - pycutest returns 0.0 at starting point
# Similar to VANDERM1 issue
class VANDERM2(AbstractConstrainedMinimisation):
    """VANDERM2 problem - Vandermonde matrix nonlinear equation system.

    A nonlinear equation problem, subject to monotonicity constraints.
    The Jacobian is a dense Vandermonde matrix.

    Problems VANDERM1, VANDERM2, VANDERM3 and VANDERM4 differ by the rhs
    of the equation. They are increasingly degenerate.

    The problem is non-convex.

    Source:
    A. Neumaier, private communication, 1991.

    SIF input: Ph. L. Toint, May 1993.
              minor correction by Ph. Shott, Jan 1995.

    Classification: NOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 100  # Number of variables (default 100)

    @property
    def m(self):
        """Number of constraints."""
        # n equality constraints (Vandermonde equations)
        # n-1 inequality constraints (monotonicity)
        return 2 * self.n - 1

    def objective(self, y, args):
        """Compute the objective (constant zero)."""
        del args, y
        # VANDERM2 has a constant zero objective
        # The equations are handled as constraints
        return jnp.array(0.0)

    def constraint(self, y):
        """Compute all constraints."""
        n = self.n
        x = y

        # Define the right-hand-side for equality constraints
        # al[i] = (1 + 1/n) - i/n for i = 1, ..., n
        i_vals = jnp.arange(1, n + 1)
        al = (1.0 + 1.0 / n) - i_vals / n

        # Compute A values
        a = jnp.zeros(n)
        a = a.at[0].set(jnp.sum(al))

        # For k >= 2, use log/exp to compute al^k to avoid numerical issues
        for k in range(2, n + 1):
            # al^k = exp(k * log(al))
            log_al = jnp.log(al)
            al_k = jnp.exp(k * log_al)
            a = a.at[k - 1].set(jnp.sum(al_k))

        # Equality constraints: the Vandermonde equations
        eq_constraints = jnp.zeros(n)

        # First equation: sum(x_i) = a[0]
        # Note: pycutest appears to use a[k] - sum(x^k) convention
        eq_constraints = eq_constraints.at[0].set(a[0] - jnp.sum(x))

        # Remaining equations: sum(x_i^k) = a[k-1]
        for k in range(2, n + 1):
            eq_constraints = eq_constraints.at[k - 1].set(a[k - 1] - jnp.sum(x**k))

        # Inequality constraints: monotonicity
        # x[i] >= x[i-1] for i = 2, ..., n
        # The SIF defines M(i) = x[i] - x[i-1] >= 0
        # But pycutest seems to return x[i] - x[i-1] directly
        ineq_constraints = x[1:] - x[:-1]

        return eq_constraints, ineq_constraints

    def y0(self):
        """Initial guess."""
        n = self.n
        # Initial point: x[i] = (i-1)/n
        return inexact_asarray(jnp.arange(n)) / n

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds (all free)."""
        n = self.n
        lower = jnp.full(n, -jnp.inf)
        upper = jnp.full(n, jnp.inf)
        return lower, upper

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
