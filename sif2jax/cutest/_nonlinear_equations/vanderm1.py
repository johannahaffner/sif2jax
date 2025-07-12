import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed - pycutest returns 0.0 at starting point
# Attempts made: Verified equations match SIF file, checked starting point formula
# Suspected issues: Possible indexing difference or different al definition in pycutest
class VANDERM1(AbstractNonlinearEquations):
    """VANDERM1 problem - Vandermonde matrix nonlinear equation system.

    A nonlinear equations problem, subject to monotonicity constraints.
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

    def num_residuals(self):
        """Number of residuals."""
        # n residual equations (Vandermonde equations)
        return self.n

    def residual(self, y, args):
        """Compute the residuals."""
        del args
        n = self.n
        x = y

        # Define the right-hand-side for residual equations
        al = inexact_asarray(jnp.arange(1, n + 1)) / n

        # Compute A values
        a = jnp.zeros(n)
        a = a.at[0].set(jnp.sum(al))

        # For k >= 2, use log/exp to compute al^k to avoid numerical issues
        for k in range(2, n + 1):
            # al^k = exp(k * log(al))
            log_al = jnp.log(al)
            al_k = jnp.exp(k * log_al)
            a = a.at[k - 1].set(jnp.sum(al_k))

        # Residual equations: the Vandermonde equations
        residuals = jnp.zeros(n)

        # First equation: sum(x_i) = a[0]
        # Note: pycutest appears to use a[k] - sum(x^k) convention
        residuals = residuals.at[0].set(a[0] - jnp.sum(x))

        # Remaining equations: sum(x_i^k) = a[k-1]
        for k in range(2, n + 1):
            residuals = residuals.at[k - 1].set(a[k - 1] - jnp.sum(x**k))

        return residuals

    def constraint(self, y):
        """Compute the constraints (both equality and inequality)."""
        # Equality constraints: the residuals
        equalities = self.residual(y, self.args())

        # Inequality constraints: monotonicity x[i] >= x[i-1] for i = 2, ..., n
        # Rewritten as x[i] - x[i-1] >= 0
        x = y
        n = self.n
        inequalities = x[1:n] - x[0 : n - 1]

        return equalities, inequalities

    def y0(self):
        """Initial guess."""
        n = self.n
        # Initial point: x[i] = (i-1)/n for i=1,...,n
        # In 0-based indexing: x[0] = 0/n, x[1] = 1/n, ..., x[n-1] = (n-1)/n
        return inexact_asarray(jnp.arange(n)) / n

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
