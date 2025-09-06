import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class REPEAT(AbstractNonlinearEquations):
    """REPEAT problem - Finding nearest feasible point to inconsistent linear equations.

    This problem is to find the nearest feasible point to 2n+1 inconsistent
    linear equations subject to bounds.

    Source: blue-cheese delerium
    SIF input: Nick Gould, December 2020.

    Classification: NLR2-AN-V-V

    Problem structure:
    - n variables
    - 2n+1 equality constraints
    - Variables subject to bounds
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 100000  # Default N value from SIF

    @property
    def m(self):
        """Number of constraints."""
        n = self.n
        return 2 * n + 1

    def constraint(self, y):
        """Compute the equality constraints (vectorized)."""
        n = self.n
        n_over_100 = n // 100

        # C constraints: C(i) = X(i) + X(i+1) - 2.0 for i=1..n-1
        #                C(n) = X(n) - 1.0
        c_constraints = jnp.zeros(n)
        c_constraints = c_constraints.at[:-1].set(y[:-1] + y[1:] - 2.0)
        c_constraints = c_constraints.at[-1].set(y[-1] - 1.0)

        # R constraints: R(i) = X(i) + X(i+1) - 4.0 for i=1..n-1
        #                R(n) = X(n) - 3.0
        r_constraints = jnp.zeros(n)
        r_constraints = r_constraints.at[:-1].set(y[:-1] + y[1:] - 4.0)
        r_constraints = r_constraints.at[-1].set(y[-1] - 3.0)

        # E constraint: sum over i: i * X(i) for i = 1, 1+n/100, ..., 1+(n-1)*n/100
        # Loop starts at 1 and increments by n/100
        indices = jnp.arange(1, n, n_over_100)
        e_constraint = jnp.sum(indices.astype(y.dtype) * y[indices - 1])

        # Combine all constraints - all are equality constraints
        equality_constraints = jnp.concatenate(
            [c_constraints, r_constraints, jnp.array([e_constraint])]
        )
        inequality_constraints = None  # No inequality constraints

        return equality_constraints, inequality_constraints

    @property
    def y0(self):
        """Initial guess."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        """Additional arguments."""
        return None

    @property
    def bounds(self):
        """Bounds on variables."""
        n = self.n
        n_half = n // 2

        # Default bounds: -1.0 <= x <= inf
        lower = jnp.full(n, -1.0)
        upper = jnp.full(n, jnp.inf)

        # Special bounds
        upper = upper.at[0].set(0.0)  # X(1) <= 0.0
        lower = lower.at[1].set(3.0)  # X(2) >= 3.0
        upper = upper.at[n_half - 1].set(0.0)  # X(n/2) <= 0.0
        lower = lower.at[n - 2].set(3.0)  # X(n-1) >= 3.0
        upper = upper.at[n - 1].set(0.0)  # X(n) <= 0.0

        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF comment: SOLTN = 0.0
        return jnp.array(0.0)
