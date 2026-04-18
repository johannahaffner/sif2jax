import jax.numpy as jnp

from ..._problem import AbstractBoundedQuadraticProblem


class NCVXBQP2(AbstractBoundedQuadraticProblem):
    """NCVXBQP2 problem - a non-convex bound constrained quadratic program.

    A non-convex bound constrained quadratic program.

    SIF input: Nick Gould, July 1995

    Classification: QBR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 10000  # Default size from SIF file

    def objective(self, y, args):
        """Compute the objective."""
        del args

        n = self.n
        x = y

        # The objective is a sum of quadratic terms
        # For each i from 1 to n:
        # OBJ(i) = 0.5 * p * (x[i] + x[mod(2i-1,n)+1] + x[mod(3i-1,n)+1])^2
        # where p = i for i <= n/2 and p = -i for i > n/2

        # Modular permutation indices (folded as constants by
        # EAGER_CONSTANT_FOLDING)
        i = jnp.arange(n)
        i2 = (2 * (i + 1) - 1) % n
        i3 = (3 * (i + 1) - 1) % n

        alpha = x + x[i2] + x[i3]

        # Weight p: +i for i < n/2, -i otherwise
        # (folded as constant by EAGER_CONSTANT_FOLDING)
        nplus = n // 2
        i_vals = jnp.arange(1, n + 1, dtype=x.dtype)
        p = jnp.where(jnp.arange(n) < nplus, i_vals, -i_vals)

        return jnp.sum(0.5 * p * alpha**2)

    @property
    def y0(self):
        """Initial guess."""
        return jnp.full(self.n, 0.5)

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return ()

    @property
    def bounds(self):
        """Variable bounds."""
        lower = jnp.full(self.n, 0.1)
        upper = jnp.full(self.n, 10.0)
        return lower, upper

    @property
    def expected_result(self):
        """Expected result based on problem name."""
        raise NotImplementedError("Expected result not available for NCVXBQP2")

    @property
    def expected_objective_value(self):
        """Expected objective value at the solution."""
        return None  # Not available
