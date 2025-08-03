import jax
import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class NCVXQP9(AbstractConstrainedQuadraticProblem):
    """NCVXQP9 problem - a non-convex quadratic program.

    A non-convex quadratic program.

    SIF input: Nick Gould, April 1995

    Classification: QLR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 10000  # Default size

    @property
    def m(self):
        """Number of constraints."""
        return (3 * self.n) // 4  # M = 3N/4

    def objective(self, y, args):
        """Compute the objective."""
        del args

        n = self.n
        x = y

        # The objective is a sum of quadratic terms
        # For each i from 1 to n:
        # OBJ(i) = 0.5 * p * (x[i] + x[mod(2i-1,n)+1] + x[mod(3i-1,n)+1])^2
        # where p = i for i <= 3n/4 and p = -i for i > 3n/4 (75% positive eigenvalues)

        def compute_term(i):
            # Positions (0-indexed)
            i1 = i
            i2 = (2 * (i + 1) - 1) % n
            i3 = (3 * (i + 1) - 1) % n

            # Sum the variables
            sum_vars = x[i1] + x[i2] + x[i3]

            # Parameter p (75% positive eigenvalues)
            nplus = (3 * n) // 4
            p = jnp.where(i < nplus, jnp.float64(i + 1), jnp.float64(-(i + 1)))

            # Compute 0.5 * p * sum^2
            return 0.5 * p * sum_vars * sum_vars

        indices = jnp.arange(n)
        terms = jax.vmap(compute_term)(indices)
        return jnp.sum(terms)

    def constraint(self, y):
        """Compute the constraints.

        Each constraint i has the form:
        x[i] + 2*x[mod(4i-1,n)+1] + 3*x[mod(5i-1,n)+1] = 6
        """
        n = self.n
        m = self.m
        x = y

        def compute_constraint(i):
            # Positions (0-indexed)
            i1 = i
            i2 = (4 * (i + 1) - 1) % n
            i3 = (5 * (i + 1) - 1) % n

            # Compute x[i] + 2*x[j] + 3*x[k] - 6
            return x[i1] + 2.0 * x[i2] + 3.0 * x[i3] - 6.0

        indices = jnp.arange(m)
        equalities = jax.vmap(compute_constraint)(indices)

        return equalities, None

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
        """Variable bounds: 0.1 <= x[i] <= 10.0 for all i."""
        n = self.n
        lower = jnp.full(n, 0.1)
        upper = jnp.full(n, 10.0)
        return lower, upper

    @property
    def expected_result(self):
        """Expected result based on problem name."""
        raise NotImplementedError("Expected result not available for NCVXQP9")

    @property
    def expected_objective_value(self):
        """Expected objective value at the solution."""
        return None  # Not available
