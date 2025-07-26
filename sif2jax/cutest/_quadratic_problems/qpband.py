import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class QPBAND(AbstractConstrainedQuadraticProblem):
    """A banded quadratic programming problem.

    SIF input: Nick Gould, December 1999.
    correction by S. Gratton & Ph. Toint, May 2024

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
        """Initial guess - default to ones."""
        return jnp.ones(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function.

        The objective has:
        - Linear terms: -i/n * x_i
        - Quadratic terms: x^T Q x where Q is tridiagonal with
          diagonal = 2, off-diagonal = -1
        """
        del args

        # Linear terms
        i_vals = jnp.arange(1, self.n + 1, dtype=jnp.float64)
        linear_term = -jnp.sum((i_vals / self.n) * y)

        # Quadratic terms: x^T Q x / 2
        # Q is tridiagonal: Q[i,i] = 2, Q[i,i+1] = Q[i+1,i] = -1
        quad_term = 0.0

        # Diagonal terms: 2 * x_i^2
        quad_term += jnp.sum(2.0 * y**2)

        # Off-diagonal terms: -2 * x_i * x_{i+1}
        quad_term += -2.0 * jnp.sum(y[:-1] * y[1:])

        return linear_term + 0.5 * quad_term

    @property
    def bounds(self):
        """Variable bounds: 0 <= x_i <= 2."""
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, 2.0)
        return lower, upper

    def constraint(self, y):
        """Linear equality constraints: x_i + x_{m+i} = 1 for i=1 to m."""
        m = self.m

        # Equality constraints: x_i + x_{m+i} - 1 = 0
        equalities = []
        for i in range(m):
            eq = y[i] + y[m + i] - 1.0
            equalities.append(eq)

        return jnp.array(equalities), None

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided."""
        return None
