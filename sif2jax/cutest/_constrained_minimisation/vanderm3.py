import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed - constraint values don't match pycutest
class VANDERM3(AbstractConstrainedMinimisation):
    """VANDERM3 problem - Vandermonde least squares with monotonicity constraints.

    A least-squares problem where we minimize the sum of squares of Vandermonde
    equation residuals, subject to monotonicity constraints.
    The Jacobian is a dense Vandermonde matrix.

    Problems VANDERM1, VANDERM2, VANDERM3 and VANDERM4 differ by the rhs
    of the equation. They are increasingly degenerate.

    The problem is non-convex.

    Source:
    A. Neumaier, private communication, 1991.

    SIF input: Ph. L. Toint, May 1993.

    Classification: NOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 100  # Number of variables (default 100)

    @property
    def m(self):
        """Number of constraints."""
        # Only n-1 inequality constraints (monotonicity)
        # The Vandermonde equations are part of the objective (L2 group type)
        return self.n - 1

    def objective(self, y, args):
        """Compute the objective (sum of squares of Vandermonde equation residuals)."""
        del args
        n = self.n
        x = y

        # Define the right-hand-side values
        al = jnp.zeros(n)
        for i in range(2, n + 1, 2):  # i = 2, 4, 6, ... (1-based)
            al = al.at[i - 2].set(i / n)  # AL(I-1) in 0-based indexing
            al = al.at[i - 1].set(i / n)  # AL(I) in 0-based indexing

        # Compute A values
        a = jnp.zeros(n)
        a = a.at[0].set(jnp.sum(al))

        # For k >= 2, use power to compute al^k
        for k in range(2, n + 1):
            # Handle zeros in al
            mask = al > 0
            al_k = jnp.where(mask, al**k, 0.0)
            a = a.at[k - 1].set(jnp.sum(al_k))

        # Compute residuals for the Vandermonde equations
        residuals = jnp.zeros(n)

        # First equation: sum(x) - a[0]
        residuals = residuals.at[0].set(jnp.sum(x) - a[0])

        # Remaining equations: sum(x^k) - a[k-1]
        for k in range(2, n + 1):
            residuals = residuals.at[k - 1].set(jnp.sum(x**k) - a[k - 1])

        # The objective is the sum of squares
        return jnp.sum(residuals**2)

    def constraint(self, y):
        """Compute all constraints."""
        x = y

        # Only inequality constraints: monotonicity
        # x[i] >= x[i-1] for i = 2, ..., n
        # The SIF defines M(i) = x[i] - x[i-1] >= 0
        ineq_constraints = x[1:] - x[:-1]

        # No equality constraints (they are in the objective as least squares)
        return None, ineq_constraints

    def y0(self):
        """Initial guess."""
        n = self.n
        # Initial point: x[i] = (i-1)/n
        return inexact_asarray(jnp.arange(n)) * n

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
        # The optimal value should be 0.0 when all equations are satisfied
        return jnp.array(0.0)
