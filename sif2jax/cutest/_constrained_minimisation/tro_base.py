"""Base class for TRO series problems with vectorized implementation."""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class TROBase(AbstractConstrainedMinimisation):
    """Base class for TRO series problems with vectorized constraint computation."""

    # These fields must be defined by subclasses
    y0_iD: int
    provided_y0s: frozenset
    m_bars: int  # number of bars
    n_dof: int  # number of structure degrees of freedom
    n: int  # total variables (m_bars + n_dof)
    m: int  # total constraints (n_dof + 1)
    obj_idx: int  # index of U variable in objective (0-based)
    rhs_idx: int  # index of constraint with RHS = 1.0 (0-based)

    @property
    def y0(self):
        # Start point: X(i) = 1/M for all bars, U(i) = 0 for all DOFs
        x_start = jnp.ones(self.m_bars) / self.m_bars
        u_start = jnp.zeros(self.n_dof)
        return jnp.concatenate([x_start, u_start])

    def objective(self, y, args):
        # Objective: minimize U at obj_idx
        return y[self.m_bars + self.obj_idx]

    def constraint(self, y):
        elements, group_uses = self.args

        # Split variables
        x = y[: self.m_bars]
        u = y[self.m_bars :]

        # Vectorized constraint computation
        # Extract arrays from elements (already jnp.array)
        x_indices, u_indices, p_values = elements.T
        x_indices = x_indices.astype(jnp.int32)
        u_indices = u_indices.astype(jnp.int32)

        # Extract constraint indices from group_uses (already jnp.array)
        c_indices, e_indices = group_uses.T
        c_indices = c_indices.astype(jnp.int32)
        e_indices = e_indices.astype(jnp.int32)

        # Compute element contributions: p * x[x_idx] * u[u_idx]
        x_vals = x[x_indices[e_indices]]
        u_vals = u[u_indices[e_indices]]
        p_vals = p_values[e_indices]
        contributions = p_vals * x_vals * u_vals

        # Accumulate contributions for each constraint
        c = jnp.zeros(self.n_dof)
        c = c.at[c_indices].add(contributions)

        # Add the balance constraint: sum(x) <= 1
        balance = jnp.sum(x) - 1.0

        # RHS for constraints: one constraint = 1.0, others = 0
        rhs = jnp.zeros(self.n_dof).at[self.rhs_idx].set(1.0)

        # Equality constraints: c - rhs = 0
        eq_constraints = c - rhs

        # Inequality constraint: balance <= 0
        ineq_constraints = jnp.array([balance])

        # Return tuple of (equalities, inequalities)
        return eq_constraints, ineq_constraints

    @property
    def bounds(self):
        # x >= 0 for design variables, u free
        lower = jnp.zeros(self.n).at[self.m_bars :].set(-jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def constraint_bounds(self):
        # First n_dof constraints are equalities (0, 0)
        # Last constraint is inequality (-inf, 0)
        lower = jnp.zeros(self.m).at[-1].set(-jnp.inf)
        upper = jnp.zeros(self.m)
        return lower, upper

    @property
    def expected_objective_value(self):
        # From the SIF file comment: SOLTN 1.0
        # This may not be the exact value, but it's a reference
        return jnp.array(1.0)

    @property
    def expected_result(self):
        # This will be determined from testing
        return None
