import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class ARGLALE(AbstractNonlinearEquations):
    """ARGLALE problem implementation.

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

    # Default parameters
    N: int = 200  # Number of variables
    M: int = 400  # Number of equations (M >= N)

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def constraint(self, y):
        """Return the equality constraints (residuals) for the nonlinear system."""
        residuals = self.residual(y)
        # Return (equality_constraints, inequality_constraints)
        # For nonlinear equations, we have only equality constraints
        return residuals, None

    def residual(self, y):
        """Compute the residuals for the system (vectorized)."""
        n = self.N
        m = self.M
        x = y  # Variables

        # Compute coefficients
        minus_two_over_m = -2.0 / m

        # Total sum of all variables
        total_sum = jnp.sum(x)

        # For the first N residuals, each residual i has:
        # G(i) = sum_{j=1}^{i-1} (-2/M) * x_j + (1-2/M) * x_i + sum_{j=i+1}^{N}
        #        (-2/M) * x_j - 1
        # = (-2/M) * [sum_{j=1}^{N} x_j - x_i] + (1-2/M) * x_i - 1
        # = (-2/M) * sum(x) + (-2/M) * (-x_i) + (1-2/M) * x_i - 1
        # = (-2/M) * sum(x) + (2/M) * x_i + x_i - (2/M) * x_i - 1
        # = (-2/M) * sum(x) + x_i - 1

        # First N residuals
        residuals_first_n = minus_two_over_m * total_sum + x - 1.0

        # Remaining M-N residuals (all identical)
        # G(i) = sum_{j=1}^{N} (-2/M) * x_j - 1 for i = N+1 to M
        remaining_value = minus_two_over_m * total_sum - 1.0
        residuals_remaining = jnp.full(m - n, remaining_value)

        # Concatenate all residuals
        residuals = jnp.concatenate([residuals_first_n, residuals_remaining])

        return residuals

    @property
    def y0(self):
        """Initial guess for variables."""
        return jnp.ones(self.N)

    @property
    def bounds(self):
        """Returns None as this problem has no bounds."""
        return None

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
        # For nonlinear equations, the objective is always zero
        return jnp.array(0.0)
