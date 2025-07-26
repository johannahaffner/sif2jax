import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class YAO(AbstractConstrainedQuadraticProblem):
    """A linear least-square problem with k-convex constraints.

    min (1/2) || f(t) - x ||^2

    subject to the constraints
    ∇^k x >= 0,

    where f(t) and x are vectors in (n+k)-dimensional space.

    We choose f(t) = sin(t), x(1) >= 0.08 and fix x(n+i) = 0

    SIF input: Aixiang Yao, Virginia Tech., May 1995
    modifications by Nick Gould

    classification QLR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    p: int = 2000  # Number of discretization points
    k: int = 2  # Degree of differences taken

    @property
    def n(self):
        """Number of variables."""
        return self.p + self.k

    @property
    def y0(self):
        """Initial guess - zeros."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function: (1/2) || f(t) - x ||^2.

        The objective is sum_i (x_i - sin(i/(p+k)))^2.
        """
        del args

        # Target function values
        i_vals = jnp.arange(1, self.n + 1, dtype=jnp.float64)
        f_vals = jnp.sin(i_vals / (self.p + self.k))

        # Objective: (1/2) sum (x_i - f_i)^2
        return 0.5 * jnp.sum((y - f_vals) ** 2)

    @property
    def bounds(self):
        """Variable bounds."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # x(1) >= 0.08
        lower = lower.at[0].set(0.08)

        # x(p+1) to x(p+k) are fixed at 0
        for i in range(self.p, self.n):
            lower = lower.at[i].set(0.0)
            upper = upper.at[i].set(0.0)

        return lower, upper

    def constraint(self, y):
        """k-convex constraints: ∇^k x >= 0.

        For k=2, this means x_i - 2*x_{i+1} + x_{i+2} >= 0 for i=1 to p.
        """
        # Inequality constraints B(i): x_i - 2*x_{i+1} + x_{i+2} >= 0
        inequalities = []
        for i in range(self.p):
            if i + 2 < self.n:
                ineq = y[i] - 2.0 * y[i + 1] + y[i + 2]
                inequalities.append(ineq)

        if len(inequalities) > 0:
            return None, jnp.array(inequalities)
        else:
            return None, None

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided."""
        return None
