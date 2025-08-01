import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class YAO(AbstractConstrainedQuadraticProblem):
    """A linear least-square problem with k-convex constraints.

    min (1/2) || f(t) - x ||^2

    subject to the constraints
    ∇^k x >= 0,

    where f(t) and x are vectors in (n+k)-dimensional space.

    We choose f(t) = sin(t), x(1) >= 0.08 and fix x(n+i) = 0

    Note: The SIF file has P+k variables, but the last k are fixed at 0.
    pycutest does NOT remove these fixed variables, so it has P+k variables.

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
        # pycutest keeps all variables including the fixed ones
        return self.p + self.k

    @property
    def y0(self):
        """Initial guess - zeros for all p+k variables."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function: (1/2) || f(t) - x ||^2.

        The objective includes all p+k terms, where the last k are fixed at 0.
        """
        del args

        # Compute f(t) = sin(t) for all p+k terms
        i_vals = jnp.arange(1, self.p + self.k + 1)
        i_vals = jnp.asarray(i_vals, dtype=y.dtype)
        f_vals = jnp.sin(i_vals / (self.p + self.k))

        # The SIF file has 'SCALE' 2.0 on each group, which means
        # each group is multiplied by 1/2
        return 0.5 * jnp.sum((y - f_vals) ** 2)

    @property
    def bounds(self):
        """Variable bounds."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # x(1) >= 0.08
        lower = lower.at[0].set(0.08)

        # The last k variables are fixed at 0
        for i in range(self.p, self.p + self.k):
            lower = lower.at[i].set(0.0)
            upper = upper.at[i].set(0.0)

        return lower, upper

    def constraint(self, y):
        """k-convex constraints: ∇^k x >= 0.

        For k=2, this means x_i - 2*x_{i+1} + x_{i+2} >= 0 for i=1 to p.
        Since pycutest keeps all variables, we can use the full vector directly.
        """
        # For constraints B(i) where i goes from 1 to P (SIF 1-based)
        # B(i): x(i) - 2*x(i+1) + x(i+2) >= 0
        # In 0-based indexing: y[i-1] - 2*y[i] + y[i+1] >= 0 for i=1 to p

        # Generate all p constraints
        # For i=0 to p-1: y[i] - 2*y[i+1] + y[i+2] >= 0
        inequalities = y[: self.p] - 2.0 * y[1 : self.p + 1] + y[2 : self.p + 2]

        return None, inequalities

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # From the SIF file: SOLUTION 1.97705D+02 (p=2000)
        if self.p == 2000:
            return jnp.array(197.705)
        return None
