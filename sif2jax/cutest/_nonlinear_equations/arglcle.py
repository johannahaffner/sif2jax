"""ARGLCLE problem.

Variable dimension rank one linear problem, with zero rows and columns

Source: Problem 34 in
J.J. More', B.S. Garbow and K.E. Hillstrom,
"Testing Unconstrained Optimization Software",
ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

See also Buckley#101 (with different N and M)
SIF input: Ph. Toint, Dec 1989.

classification NLR2-AN-V-V

This is a(n infeasible) linear feasibility problem 
N is the number of free variables
M is the number of equations ( M.ge.N)
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class ARGLCLE(AbstractUnconstrainedMinimisation):
    """ARGLCLE problem implementation."""

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

        # G(1) - first equation has no variables (empty row)
        residuals.append(-1.0)

        # G(i) for i = 2 to M-1
        for i in range(2, m):
            # G(i) = sum_{j=2}^{N-1} (i-1) * j * x_j - 1
            res = 0.0
            for j in range(2, n):
                res += float((i - 1) * j) * x[j - 1]  # Convert to 0-based indexing
            res -= 1.0
            residuals.append(res)

        # G(M) - last equation has no variables (empty row)
        residuals.append(-1.0)

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
        # SOLTN(10)  = 6.13513513
        # SOLTN(50)  = 26.1269035
        # SOLTN(100) = 26.1269
        return None

    @property
    def n(self):
        """Number of variables."""
        return self.N

    @property
    def m(self):
        """Number of equations/residuals."""
        return self.M
