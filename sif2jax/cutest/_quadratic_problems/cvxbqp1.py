import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractBoundedQuadraticProblem


class CVXBQP1(AbstractBoundedQuadraticProblem):
    """CVXBQP1 problem - a convex bound constrained quadratic program.

    A convex bound constrained quadratic program.

    SIF input: Nick Gould, July 1995

    Classification: QBR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 100000  # Default size from SIF file

    def objective(self, y, args):
        """Compute the objective."""
        del args

        n = self.n
        x = y

        # The objective is a sum of quadratic terms
        # For each i from 1 to n:
        # OBJ(i) = 0.5 * i * (x[i] + x[mod(2i-1,n)+1] + x[mod(3i-1,n)+1])^2

        def compute_term(i):
            # Positions (0-indexed)
            i1 = i

            # For mod(2i-1, n) + 1 in SIF notation:
            # i is 0-indexed here, so (i+1) is 1-indexed
            i2 = (2 * (i + 1) - 1) % n  # This gives 0-indexed position

            # For mod(3i-1, n) + 1 in SIF notation:
            i3 = (3 * (i + 1) - 1) % n  # This gives 0-indexed position

            alpha = x[i1] + x[i2] + x[i3]
            p = inexact_asarray(i + 1)  # P parameter is i (1-indexed)

            return 0.5 * p * alpha * alpha

        # Use vmap to vectorize over all indices
        terms = jax.vmap(compute_term)(jnp.arange(n))
        obj = jnp.sum(terms)

        return obj

    @property
    def y0(self):
        """Initial guess."""
        # Default value is 0.5
        return inexact_asarray(jnp.full(self.n, 0.5))

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Variable bounds."""
        # 0.1 <= x[i] <= 10.0 for all i
        lower = jnp.full(self.n, 0.1)
        upper = jnp.full(self.n, 10.0)
        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From comment in SIF file for n=100
        # Solution: 2.27250D+02
        if self.n == 100:
            return jnp.array(227.250)
        return None
