import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class QPNBAND(AbstractConstrainedQuadraticProblem):
    """A banded non-convex quadratic programming problem.

    Source: N. I. M. Gould, "An algorithm for large-scale quadratic programming",
    IMA J. Num. Anal (1991), 11, 299-324, problem class 4.

    SIF input: Nick Gould, January 2000.

    classification QLR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 50000  # Default problem size

    @property
    def m(self):
        """Number of constraints = n/2."""
        return self.n // 2

    @property
    def y0(self):
        """Initial guess - default to zeros."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Non-convex banded quadratic objective function.

        The objective has:
        - Linear terms: -i/n * x_i
        - Quadratic terms: tridiagonal structure from QUADRATIC section
        """
        del args

        # Linear terms: -i/n * x_i
        i_vals = jnp.arange(1, self.n + 1)
        i_vals = jnp.asarray(i_vals, dtype=y.dtype)
        linear_term = -jnp.sum((i_vals / self.n) * y)

        # Quadratic terms from QUADRATIC section:
        # For i=1 to M: X(I) X(I) -2.0  X(I+1) -1.0
        # For i=M+1 to N-1: X(I) X(I) 2.0  X(I+1) -1.0
        # For i=N: X(N) X(N) 2.0

        m = self.m
        quad_term = 0.0

        # Diagonal terms
        # i=1 to M: -2.0 * x_i^2
        quad_term += -2.0 * jnp.sum(y[:m] ** 2)
        # i=M+1 to N-1: 2.0 * x_i^2
        quad_term += 2.0 * jnp.sum(y[m : self.n - 1] ** 2)
        # i=N: 2.0 * x_N^2
        quad_term += 2.0 * y[self.n - 1] ** 2

        # Off-diagonal terms (multiplied by 2 since H is symmetric)
        # i=1 to M: 2 * (-1.0) * x_i * x_{i+1}
        quad_term += 2.0 * (-1.0) * jnp.sum(y[:m] * y[1 : m + 1])
        # i=M+1 to N-1: 2 * (-1.0) * x_i * x_{i+1}
        quad_term += 2.0 * (-1.0) * jnp.sum(y[m : self.n - 1] * y[m + 1 : self.n])

        return linear_term + 0.5 * quad_term

    @property
    def bounds(self):
        """Variable bounds: 0 <= x_i <= 2."""
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, 2.0)
        return lower, upper

    def constraint(self, y):
        """Linear inequality constraints: x_i + x_{m+i} >= 1 for i=1 to m."""
        m = self.m

        # Inequality constraints: x_i + x_{m+i} >= 1
        # Convert to form g(x) >= 0: x_i + x_{m+i} - 1 >= 0
        inequalities = y[:m] + y[m : 2 * m] - 1.0

        return None, inequalities

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided."""
        return None
