import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class CVXQP1(AbstractConstrainedMinimisation):
    """CVXQP1 problem - a convex quadratic program.

    A convex quadratic program with sparse structure.

    SIF input: Nick Gould, May 1995

    Classification: QLR2-AN-V-V
    """

    @property
    def n(self):
        """Number of variables."""
        return 100  # Default size

    @property
    def m(self):
        """Number of constraints."""
        return self.n // 2  # M = N/2

    def objective(self, y, args):
        """Compute the objective."""
        del args

        n = self.n
        x = y

        # The objective is a sum of quadratic terms
        # For each i from 1 to n:
        # OBJ(i) = 0.5 * i * (x[i] + x[mod(2i-1, n)] + x[mod(3i-1, n)])^2

        obj = 0.0
        for i in range(n):
            # Positions (0-indexed)
            i1 = i
            i2 = (2 * (i + 1) - 1 - 1) % n  # mod(2i-1, n) in 0-indexed
            i3 = (3 * (i + 1) - 1 - 1) % n  # mod(3i-1, n) in 0-indexed

            alpha = x[i1] + x[i2] + x[i3]
            p = float(i + 1)  # P parameter is i (1-indexed)

            obj += 0.5 * p * alpha * alpha

        return jnp.array(obj)

    def constraint(self, y):
        """Compute the constraints."""
        n = self.n
        m = self.m
        x = y

        # Linear equality constraints
        # For each i from 1 to m:
        # CON(i) = x[i] + 2*x[mod(4i-1, n)] + 3*x[mod(5i-1, n)] = 6

        constraints = []
        for i in range(m):
            # Positions (0-indexed)
            i1 = i
            i2 = (4 * (i + 1) - 1 - 1) % n  # mod(4i-1, n) in 0-indexed
            i3 = (5 * (i + 1) - 1 - 1) % n  # mod(5i-1, n) in 0-indexed

            c = x[i1] + 2.0 * x[i2] + 3.0 * x[i3] - 6.0
            constraints.append(c)

        return jnp.array(constraints), None

    def equality_constraints(self):
        """All constraints are equalities."""
        return jnp.ones(self.m, dtype=bool)

    def y0(self):
        """Initial guess."""
        # Default value is 0.5
        return jnp.full(self.n, 0.5)

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds."""
        # 0.1 <= x[i] <= 10.0 for all i
        lower = jnp.full(self.n, 0.1)
        upper = jnp.full(self.n, 10.0)
        return lower, upper

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From comment in SIF file for n=1000
        # We don't have the value for n=100
        return None
