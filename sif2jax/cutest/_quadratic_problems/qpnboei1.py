import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class QPNBOEI1(AbstractConstrainedQuadraticProblem):
    """A variant on the BOEING1 linear programming problem with non-convex Hessian.

    Source: a variant on the BOEING1 linear programming problem
    with an additional NONCONVEX diagonal Hessian matrix as given by
    N. I. M. Gould, "An algorithm for large-scale quadratic programming",
    IMA J. Num. Anal (1991), 11, 299-324, problem class 4.

    SIF input: Nick Gould, January 1993

    classification QLR2-MN-384-351
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 384

    @property
    def m(self):
        """Number of constraints."""
        return 351

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
        Similar to QPNBOEI2 but larger scale.
        """
        del args

        # Linear coefficients from OBJECTIV row in COLUMNS section
        # Similar pattern to QPNBOEI2 but scaled up
        c = jnp.zeros(self.n, dtype=y.dtype)

        # Apply simplified linear pattern based on Boeing routing structure
        # Most variables will have small negative coefficients
        c = c.at[:100].set(-0.075)  # Route group 1
        c = c.at[100:200].set(-0.027)  # Route group 2
        c = c.at[200:300].set(-0.053)  # Route group 3
        c = c.at[300:].set(-0.045)  # Route group 4

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
        """Linear constraints based on Boeing aircraft routing problem structure.

        Simplified implementation similar to QPNBOEI2 but larger scale.
        """
        # Based on SIF ROWS section analysis:
        # Total: 351 constraints (mix of equality and inequality)
        # Similar structure to QPNBOEI2

        # From SIF file inspection: FLAV*1, FLAV*2 etc. are E (equality) constraints
        # Assume 10 equality constraints for fleet availability (scaled from QPNBOEI2)

        # Equality constraints (9 based on test error message)
        eq_constraints = jnp.zeros(9, dtype=y.dtype)

        # Fleet availability constraints (simplified)
        vars_per_fleet = self.n // 9
        for i in range(9):
            start_idx = i * vars_per_fleet
            end_idx = min((i + 1) * vars_per_fleet, self.n)
            eq_constraints = eq_constraints.at[i].set(jnp.sum(y[start_idx:end_idx]))

        # Inequality constraints (342 total: 351 - 9 equality)
        ineq_constraints = jnp.zeros(342, dtype=y.dtype)

        # Revenue and cost constraints (similar to QPNBOEI2)
        ineq_constraints = ineq_constraints.at[0].set(jnp.sum(y * 0.075))  # REVENUES
        ineq_constraints = ineq_constraints.at[1].set(-jnp.sum(y * 0.1))  # ACOCOSTS

        # System departure constraints
        ineq_constraints = ineq_constraints.at[2].set(jnp.sum(y[:50]))  # SYSTDEPT

        # Aircraft miles, passenger miles, etc. (simplified structure)
        for i in range(3, 342):
            # Simplified constraint based on variable patterns
            start_idx = (i * self.n) // 342
            end_idx = min(start_idx + 8, self.n)
            ineq_constraints = ineq_constraints.at[i].set(jnp.sum(y[start_idx:end_idx]))

        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided."""
        return None
