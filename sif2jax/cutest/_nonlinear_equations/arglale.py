"""ARGLALE problem.

Variable dimension full rank linear problem, a linear equation version.

Source: Problem 32 in
J.J. More', B.S. Garbow and K.E. Hillstrom,
"Testing Unconstrained Optimization Software",
ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

See also Buckley#80 (with different N and M)
SIF input: Ph. Toint, Dec 1989.

classification NLR2-AN-V-V

This is a(n infeasible) linear feasibility problem 
N is the number of free variables
M is the number of equations ( M.ge.N)
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class ARGLALE(AbstractUnconstrainedMinimisation):
    """ARGLALE problem implementation."""

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

        # Compute -2/M
        minus_two_over_m = -2.0 / m
        one_minus_two_over_m = 1.0 + minus_two_over_m

        # Initialize residual array
        residuals = []

        # First N residuals (i = 1 to N)
        for i in range(n):
            # G(i) = sum_{j=1}^{i-1} (-2/M) * x_j + (1-2/M) * x_i
            #        + sum_{j=i+1}^{N} (-2/M) * x_j - 1
            res = 0.0
            # Sum over j < i
            if i > 0:
                res += minus_two_over_m * jnp.sum(x[:i])
            # Diagonal term
            res += one_minus_two_over_m * x[i]
            # Sum over j > i
            if i < n - 1:
                res += minus_two_over_m * jnp.sum(x[i + 1 :])
            # Subtract constant
            res -= 1.0
            residuals.append(res)

        # Remaining M-N residuals (i = N+1 to M)
        for i in range(n, m):
            # G(i) = sum_{j=1}^{N} (-2/M) * x_j - 1
            res = minus_two_over_m * jnp.sum(x) - 1.0
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
        return None

    @property
    def n(self):
        """Number of variables."""
        return self.N

    @property
    def m(self):
        """Number of equations/residuals."""
        return self.M
