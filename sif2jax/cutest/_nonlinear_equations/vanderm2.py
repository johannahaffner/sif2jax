import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed - pycutest returns 0.0 at starting point
# Similar to VANDERM1 issue
class VANDERM2(AbstractNonlinearEquations):
    """VANDERM2 problem - Vandermonde matrix nonlinear equation system.

    A nonlinear equations problem, subject to monotonicity constraints.
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
        # al[i] = (1 + 1/n) - i/n for i = 1, ..., n
        i_vals = jnp.arange(1, n + 1)
        al = (1.0 + 1.0 / n) - i_vals / n

        # Compute A values vectorized
        # Create k values from 1 to n
        k_values = jnp.arange(1, n + 1, dtype=float)[:, None]  # Shape: (n, 1)

        # Compute al^k for all k using broadcasting with log/exp for numerical stability
        # Note: for k=1, al^1 = al
        log_al = jnp.log(al[None, :])  # Shape: (1, n)
        al_powers = jnp.exp(k_values * log_al)  # Shape: (n, n)

        # Sum over al dimension to get a[k-1] for k=1,2,...,n
        a = jnp.sum(al_powers, axis=1)  # Shape: (n,)

        # Residual equations: the Vandermonde equations
        # Compute x^k for k=1,2,...,n vectorized
        x_powers = x[None, :] ** k_values  # Shape: (n, n)

        # Sum over x dimension to get sum(x^k) for all k
        x_sums = jnp.sum(x_powers, axis=1)  # Shape: (n,)

        # Compute residuals: a[k-1] - sum(x^k)
        # Note: pycutest appears to use a[k] - sum(x^k) convention
        residuals = a - x_sums

        return residuals

    @property
    def y0(self):
        """Initial guess."""
        n = self.n
        # Initial point: x[i] = (i-1)/n
        return inexact_asarray(jnp.arange(n)) / n

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
