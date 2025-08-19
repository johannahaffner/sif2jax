"""DEGENQPC problem from CUTEst collection.

Classification: QLR2-AN-V-V

A simple degenerate convex quadratic program with a large number of constraints.
Corrected version of DEGENQP.

Source: Nick Gould, March 2013

SIF input: Nick Gould, March 2013
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class DEGENQPC(AbstractConstrainedQuadraticProblem):
    """DEGENQPC problem from CUTEst collection.

    Quadratic programming problem with O(N^3) constraints.
    Corrected version of DEGENQP.
    Fixed size: N=50 (default from SIF)
    """

    n: int = 50  # Number of variables
    m_eq: int = 25  # Number of equality constraints (n // 2)
    m_ineq: int = 19600  # Number of inequality constraints (C(n,3))
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Initial guess - all variables set to 2.0."""
        return jnp.full(self.n, 2.0)

    @property
    def bounds(self):
        """Variable bounds: 0 <= x_i <= 1."""
        lower = jnp.zeros(self.n)
        upper = jnp.ones(self.n)
        return lower, upper

    def objective(self, y, args):
        """Quadratic objective function.

        f(x) = sum_i (i/n) * x_i^2 / 2 + x_i
        """
        del args
        n = self.n
        indices = jnp.arange(1, n + 1)
        coeffs = indices / n

        # Quadratic term: (i/n) * x_i^2 / 2
        quad_term = 0.5 * jnp.sum(coeffs * y**2)

        # Linear term: sum(x_i)
        linear_term = jnp.sum(y)

        return quad_term + linear_term

    def constraint(self, y):
        """Returns the constraints on the variable y.

        Equality constraints: x_i - x_{i+1} = 0 for odd i
        Inequality constraints: 0 <= x_i + x_j + x_k <= 2 for i < j < k
        """
        n = self.n

        # Equality constraints
        eq_constraints = jnp.zeros(self.m_eq)
        eq_idx = 0
        for i in range(
            0, n - 1, 2
        ):  # i = 0, 2, 4, ... (corresponding to x_1, x_3, x_5, ...)
            eq_constraints = eq_constraints.at[eq_idx].set(y[i] - y[i + 1])
            eq_idx += 1

        # Inequality constraints (range constraints)
        # For each triple (i,j,k) with i < j < k:
        # Range constraint 0 <= x_i + x_j + x_k <= 2
        # CUTEst treats these as single inequality constraints: x_i + x_j + x_k >= 0
        ineq_constraints = jnp.zeros(self.m_ineq)
        ineq_idx = 0

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    sum_ijk = y[i] + y[j] + y[k]
                    # Only the lower bound: sum >= 0
                    ineq_constraints = ineq_constraints.at[ineq_idx].set(sum_ijk)
                    ineq_idx += 1

        return eq_constraints, ineq_constraints

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Optimal solution not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Optimal objective value not provided in SIF file."""
        return None
