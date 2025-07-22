import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed - constraint values don't match pycutest
class VANDERM3(AbstractNonlinearEquations):
    """VANDERM3 problem - Vandermonde matrix nonlinear equation system.

    A nonlinear equations problem, subject to monotonicity constraints. The Jacobian is
    a dense Vandermonde matrix.

    Problems VANDERM1, VANDERM2, VANDERM3 and VANDERM4 differ by the right hand side
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

        return residuals

    @property
    def y0(self):
        """Initial guess."""
        n = self.n
        # Initial point: x[i] = (i-1)/n
        return inexact_asarray(jnp.arange(n)) * n

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # The optimal value should be 0.0 when all equations are satisfied
        return jnp.array(0.0)

    def constraint(self, y):
        """Compute the constraints (both equality and inequality)."""
        # Equality constraints: the residuals
        equalities = self.residual(y, self.args)

        # Inequality constraints: monotonicity x[i] >= x[i-1] for i = 2, ..., n
        # Rewritten as x[i] - x[i-1] >= 0
        x = y
        n = self.n
        inequalities = x[1:n] - x[0 : n - 1]

        return equalities, inequalities

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        """No bounds for this problem."""
        return None
