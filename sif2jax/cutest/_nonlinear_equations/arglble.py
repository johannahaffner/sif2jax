"""ARGLBLE problem.

Variable dimension rank one linear problem

Source: Problem 33 in
J.J. More', B.S. Garbow and K.E. Hillstrom,
"Testing Unconstrained Optimization Software",
ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

See also Buckley#93 (with different N and M)
SIF input: Ph. Toint, Dec 1989.

classification NLR2-AN-V-V

This is a(n infeasible) linear feasibility problem
N is the number of free variables
M is the number of equations ( M .ge. N)
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class ARGLBLE(AbstractUnconstrainedMinimisation):
    """ARGLBLE problem implementation."""

    # Default parameters
    N: int = 200  # Number of variables
    M: int = 400  # Number of equations (M >= N)

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Compute the objective function."""
        del args

        # This is formulated as a nonlinear least squares problem
        # Compute the residuals
        residuals = self.residual(y)

        # Sum of squares
        return 0.5 * jnp.sum(residuals**2)

    def residual(self, y):
        """Compute the residuals for the system."""
        n = self.N
        m = self.M
        x = y  # Variables

        # Initialize residual array
        residuals = []

        # For each equation i = 1 to M
        for i in range(1, m + 1):
            # G(i) = sum_{j=1}^{N} (i * j) * x_j - 1
            res = 0.0
            for j in range(1, n + 1):
                res += float(i * j) * x[j - 1]  # Convert to 0-based indexing
            res -= 1.0
            residuals.append(res)

        return jnp.array(residuals)

    @property
    def y0(self):
        """Initial guess for variables."""
        return jnp.ones(self.N)

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file:
        # SOLTN(10)  = 4.6341D+00
        # SOLTN(50)  = 24.6268657
        # SOLTN(100) = 49.6259352
        return None

    @property
    def n(self):
        """Number of variables."""
        return self.N

    @property
    def m(self):
        """Number of equations/residuals."""
        return self.M
