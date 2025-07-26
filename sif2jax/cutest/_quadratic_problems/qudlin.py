import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class QUDLIN(AbstractConstrainedQuadraticProblem):
    """A simple quadratic programming problem.

    The objective consists of linear terms and product terms x_i * x_{i+1}.

    SIF input: unknown.
    minor correction by Ph. Shott, Jan 1995.

    classification QBR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000  # Number of variables
    m: int = 2500  # Number of product terms

    @property
    def y0(self):
        """Initial guess - zeros."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function.

        Linear terms: (i-10) * x_i for i=1 to n
        Quadratic terms: x_i * x_{i+1} for i=1 to m
        """
        del args

        # Linear terms
        i_vals = jnp.arange(1, self.n + 1, dtype=jnp.float64)
        coeffs = i_vals - 10.0
        linear_term = jnp.dot(coeffs, y)

        # Quadratic terms: sum of x_i * x_{i+1} for i=1 to m
        quad_term = 0.0
        if self.m > 0 and self.n > 1:
            # Ensure we don't exceed array bounds
            max_idx = min(self.m, self.n - 1)
            quad_term = jnp.sum(y[:max_idx] * y[1 : max_idx + 1])

        return linear_term + quad_term

    @property
    def bounds(self):
        """Variable bounds: 0 <= x_i <= 10."""
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, 10.0)
        return lower, upper

    def constraint(self, y):
        """No additional constraints beyond bounds."""
        return None, None

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value is 0.0 (from SIF file)."""
        return jnp.array(0.0)
