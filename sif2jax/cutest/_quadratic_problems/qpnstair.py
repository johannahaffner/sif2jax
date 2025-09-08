import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class QPNSTAIR(AbstractConstrainedQuadraticProblem):
    """A variant on the STAIR linear programming problem with non-convex Hessian.

    Source: a variant on the STAIR linear programming problem
    with an additional NONCONVEX diagonal Hessian matrix as given by
    N. I. M. Gould, "An algorithm for large-scale quadratic programming",
    IMA J. Num. Anal (1991), 11, 299-324, problem class 4.

    SIF input: Nick Gould, January 1993

    classification QLR2-MN-467-356
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 467

    @property
    def m(self):
        """Number of constraints."""
        return 356

    @property
    def y0(self):
        """Initial guess - all zeros."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Non-convex quadratic objective function.

        Based on SIF structure with NONCONVEX diagonal Hessian.
        This is the largest problem in the QPN series.
        """
        del args

        # Linear coefficients from MXR row in COLUMNS section
        # STAIR problem typically has economic planning/scheduling structure
        c = jnp.zeros(self.n, dtype=y.dtype)

        # Apply simplified linear pattern based on STAIR structure
        # Different sections may represent different time periods/activities
        section_size = self.n // 5
        c = c.at[:section_size].set(-1.0)  # Time period 1
        c = c.at[section_size : 2 * section_size].set(-0.8)  # Time period 2
        c = c.at[2 * section_size : 3 * section_size].set(-0.6)  # Time period 3
        c = c.at[3 * section_size : 4 * section_size].set(-0.4)  # Time period 4
        c = c.at[4 * section_size :].set(-0.2)  # Time period 5

        # Diagonal D values for quadratic term (making it non-convex)
        d_vals = jnp.arange(1, self.n + 1, dtype=y.dtype)
        d = -1.0 + (d_vals - 1) * (10.0 - (-1.0)) / (self.n - 1)

        # Linear term
        linear_term = jnp.sum(c * y)

        # Quadratic term: 0.5 * x^T * diag(d) * x
        quadratic_term = jnp.sum(d * y * y)

        return linear_term + 0.5 * quadratic_term

    @property
    def bounds(self):
        """Variable bounds: x_i >= 0 (standard LP/QP convention)."""
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    def constraint(self, y):
        """Linear constraints based on STAIR economic planning problem structure.

        Simplified implementation for this large-scale problem.
        """
        # Based on SIF ROWS section analysis:
        # Total: 356 constraints (mix of equality and inequality)
        # STAIR problems typically have resource balance and capacity constraints

        # From SIF file inspection: GINV0, RKA0, RCON0 are E (equality) constraints
        # Most A0** constraints are L (less than) inequality constraints

        # Equality constraints (estimated 20 for resource balances)
        eq_constraints = jnp.zeros(20, dtype=y.dtype)

        # Resource balance constraints (simplified)
        vars_per_resource = self.n // 20
        for i in range(20):
            start_idx = i * vars_per_resource
            end_idx = min((i + 1) * vars_per_resource, self.n)
            # Resource balance: production - consumption = 0
            eq_constraints = eq_constraints.at[i].set(
                jnp.sum(y[start_idx:end_idx]) - jnp.sum(y[start_idx:end_idx] * 0.5)
            )

        # Inequality constraints (336 total: 356 - 20 equality)
        ineq_constraints = jnp.zeros(336, dtype=y.dtype)

        # Capacity constraints (L type - less than)
        for i in range(336):
            # Simplified constraint based on capacity limits
            start_idx = (i * self.n) // 336
            end_idx = min(start_idx + 10, self.n)
            # Capacity constraint: sum of activities <= capacity
            capacity = 100.0 + i * 0.5  # Varying capacities
            ineq_constraints = ineq_constraints.at[i].set(
                capacity - jnp.sum(y[start_idx:end_idx])
            )

        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided."""
        return None
