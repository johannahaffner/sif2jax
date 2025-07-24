import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class VARDIM(AbstractUnconstrainedMinimisation):
    """Variable dimension least squares problem.

    This problem is a sum of n+2 least-squares groups, the first n of
    which have only a linear element. Its Hessian matrix is dense.

    Source: problem 25 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#72 (p.98).

    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem dimension
    N: int = 200  # Default from SIF file

    @property
    def n(self):
        """Number of variables."""
        return self.N

    def objective(self, y, args):
        """Compute the objective function as sum of squares.

        The objective is:
        sum_{i=1}^n (x_i - 1)^2 + (sum_{i=1}^n i*x_i - n*(n+1)/2)^2 +
        (sum_{i=1}^n i*x_i - n*(n+1)/2)^4
        """
        del args

        # Compute n*(n+1)/2
        n_times_n_plus_1_over_2 = self.N * (self.N + 1) / 2.0

        # Linear terms sum_i i*x_i
        indices = jnp.arange(1, self.N + 1, dtype=y.dtype)
        linear_sum = jnp.sum(indices * y)

        # First n groups: (x_i - 1)^2
        first_n_groups = jnp.sum((y - 1.0) ** 2)

        # Group n+1: (sum_i i*x_i - n*(n+1)/2)^2
        group_n_plus_1 = (linear_sum - n_times_n_plus_1_over_2) ** 2

        # Group n+2: (sum_i i*x_i - n*(n+1)/2)^4
        group_n_plus_2 = (linear_sum - n_times_n_plus_1_over_2) ** 4

        return first_n_groups + group_n_plus_1 + group_n_plus_2

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
    def expected_result(self):
        """Expected optimal solution is x_i = 1 for all i."""
        return jnp.ones(self.N)

    @property
    def expected_objective_value(self):
        """Expected optimal objective value is 0."""
        return jnp.array(0.0)
