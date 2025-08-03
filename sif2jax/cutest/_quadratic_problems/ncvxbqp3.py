import jax
import jax.numpy as jnp

from ..._problem import AbstractBoundedQuadraticProblem


class NCVXBQP3(AbstractBoundedQuadraticProblem):
    """NCVXBQP3 problem - a non-convex bound constrained quadratic program.

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
        # where p = i for i <= 3n/4 and p = -i for i > 3n/4

        def compute_term(i):
            # Positions (0-indexed)
            i1 = i
            i2 = (2 * (i + 1) - 1) % n
            i3 = (3 * (i + 1) - 1) % n

            # Sum the variables
            sum_vars = x[i1] + x[i2] + x[i3]

            # Parameter p (75% positive eigenvalues)
            nplus = (n // 4) * 3  # 3n/4
            p = jnp.where(i < nplus, jnp.float64(i + 1), jnp.float64(-(i + 1)))

            # Compute 0.5 * p * sum^2
            return 0.5 * p * sum_vars * sum_vars

        indices = jnp.arange(n)
        terms = jax.vmap(compute_term)(indices)
        return jnp.sum(terms)

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
        raise NotImplementedError("Expected result not available for NCVXBQP3")

    @property
    def expected_objective_value(self):
        """Expected objective value at the solution."""
        return None  # Not available
