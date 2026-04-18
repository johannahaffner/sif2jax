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

        # OBJ(i) = 0.5 * i * (x[i] + x[mod(2i-1,n)+1] + x[mod(3i-1,n)+1])^2
        # Modular permutation indices (folded as constants by
        # EAGER_CONSTANT_FOLDING)
        i = jnp.arange(n)
        i2 = (2 * (i + 1) - 1) % n
        i3 = (3 * (i + 1) - 1) % n

        # Identity permutation is just x itself
        alpha = x + x[i2] + x[i3]
        p = jnp.arange(1, n + 1, dtype=x.dtype)

        return jnp.sum(0.5 * p * alpha**2)

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
