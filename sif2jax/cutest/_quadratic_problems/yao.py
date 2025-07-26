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
    pycutest removes these fixed variables, so it only has P variables.

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
        # pycutest removes the k fixed variables, so n = p
        return self.p

    @property
    def y0(self):
        """Initial guess - zeros."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function: (1/2) || f(t) - x ||^2.

        The objective includes all p+k terms, where the last k are fixed at 0.
        """
        del args

        # Contribution from the p free variables
        i_vals_free = jnp.arange(1, self.p + 1)
        i_vals_free = jnp.asarray(i_vals_free, dtype=y.dtype)
        f_vals_free = jnp.sin(i_vals_free / (self.p + self.k))
        contrib_free = jnp.sum((y - f_vals_free) ** 2)

        # Contribution from the k fixed variables (fixed at 0)
        # These are x_{p+1} = 0, ..., x_{p+k} = 0
        i_vals_fixed = jnp.arange(self.p + 1, self.p + self.k + 1)
        i_vals_fixed = jnp.asarray(i_vals_fixed, dtype=y.dtype)
        f_vals_fixed = jnp.sin(i_vals_fixed / (self.p + self.k))
        contrib_fixed = jnp.sum((0.0 - f_vals_fixed) ** 2)

        # The SIF file has 'SCALE' 2.0 on each group, which means
        # each group is multiplied by 1/2
        return 0.5 * (contrib_free + contrib_fixed)

    @property
    def bounds(self):
        """Variable bounds."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # x(1) >= 0.08
        lower = lower.at[0].set(0.08)

        # Note: The last k variables in the SIF file are fixed at 0,
        # but pycutest removes them, so we don't need to handle them

        return lower, upper

    def constraint(self, y):
        """k-convex constraints: ∇^k x >= 0.

        For k=2, this means x_i - 2*x_{i+1} + x_{i+2} >= 0 for i=1 to p.
        Since pycutest removes the fixed variables, we need to handle the
        constraints that involve them by treating those variables as 0.
        """
        p = self.p

        # For constraints B(i) where i goes from 1 to P (SIF 1-based)
        # B(i): x(i) - 2*x(i+1) + x(i+2) >= 0

        # Most constraints use three consecutive variables
        # For i=1 to p-2: normal constraints with all variables present
        if p > 2:
            # Standard constraints for i in range(p-2)
            inequalities = y[:-2] - 2.0 * y[1:-1] + y[2:]

            # For the last two constraints (i=p-1 and i=p), we need the fixed variables
            # B(p-1): x(p-1) - 2*x(p) + x(p+1) >= 0, where x(p+1)=0
            constraint_p_minus_1 = y[-2] - 2.0 * y[-1] + 0.0

            # B(p): x(p) - 2*x(p+1) + x(p+2) >= 0, where x(p+1)=0 and x(p+2)=0
            constraint_p = y[-1] - 2.0 * 0.0 + 0.0

            # Concatenate all constraints
            all_inequalities = jnp.concatenate(
                [inequalities, jnp.array([constraint_p_minus_1, constraint_p])]
            )

            return None, all_inequalities
        else:
            # Handle edge case for very small p
            return None, jnp.array([])

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
