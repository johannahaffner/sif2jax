import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class VARDIMNE(AbstractNonlinearEquations):
    """Variable dimension nonlinear equations problem.

    This problem is a sum of n+2 least-squares groups, the first n of
    which have only a linear element. Its Hessian matrix is dense.
    This is a nonlinear equation version of problem VARDIM.

    Source: problem 25 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#72 (p.98).

    SIF input: Ph. Toint, Dec 1989.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    Classification: NOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimension
    N: int = 10  # Default from SIF file

    @property
    def n(self):
        """Number of variables."""
        return self.N

    @property
    def m(self):
        """Number of equations."""
        return self.N + 2

    def residual(self, y):
        """Compute the residuals for the nonlinear equations.

        The residuals are:
        - First n equations: x_i - 1 for i=1..n
        - Equation n+1: sum_{i=1}^n i*x_i - n*(n+1)/2
        - Equation n+2: (sum_{i=1}^n i*x_i - n*(n+1)/2)^2
        """
        # Compute n*(n+1)/2
        n_times_n_plus_1_over_2 = self.N * (self.N + 1) / 2.0

        # Linear terms sum_i i*x_i
        indices = jnp.arange(1, self.N + 1, dtype=y.dtype)
        linear_sum = jnp.sum(indices * y)

        # First n residuals: x_i - 1
        first_n_residuals = y - 1.0

        # Residual n+1: sum_i i*x_i - n*(n+1)/2
        residual_n_plus_1 = linear_sum - n_times_n_plus_1_over_2

        # Residual n+2: (sum_i i*x_i - n*(n+1)/2)^2
        # Note: This is the squared version for the nonlinear equation
        residual_n_plus_2 = residual_n_plus_1**2

        # Combine all residuals
        residuals = jnp.concatenate(
            [
                first_n_residuals,
                jnp.array([residual_n_plus_1]),
                jnp.array([residual_n_plus_2]),
            ]
        )

        return residuals

    def constraint(self, y):
        """Return the equality constraints (residuals) for the nonlinear system."""
        residuals = self.residual(y)
        # Return (equality_constraints, inequality_constraints)
        # For nonlinear equations, we have only equality constraints
        return residuals, None

    @property
    def y0(self):
        """Initial guess: x_i = 1 - i/n."""
        indices = jnp.arange(1, self.N + 1, dtype=jnp.float64)
        return 1.0 - indices / self.N

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """No bounds for this problem."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution is x_i = 1 for all i."""
        return jnp.ones(self.N)

    @property
    def expected_objective_value(self):
        """Expected optimal objective value is 0."""
        return jnp.array(0.0)
