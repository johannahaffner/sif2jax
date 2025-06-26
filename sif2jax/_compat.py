"""Compatibility wrapper for Optimistix optimization library."""

from typing import Any

import jax.flatten_util as jfu
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Int, PyTree, Scalar

from ._problem import AbstractConstrainedMinimisation


class OptimistixWrapper(AbstractConstrainedMinimisation):
    """Compatibility wrapper that transforms constraints from sif2jax/pycutest
    convention (bounded format: cl ≤ c(x) ≤ cu) to Optimistix convention
    (g(x) ≥ 0, h(x) = 0).

    This wrapper handles the following transformations:
    - Equality constraints (where cl = cu = 0): Passed through as h(x) = 0
    - G-type inequalities (cl > -inf, cu = inf): Convert to c(x) - cl ≥ 0
    - L-type inequalities (cl = -inf, cu < inf): Convert to -c(x) + cu ≥ 0
    - Range constraints (finite cl and cu): Split into two inequalities

    The wrapped problem must implement:
    - constraint(): Returns (equality_values, inequality_values)
    - constraint_bounds(): Returns (lower_bounds, upper_bounds)

    This wrapper is designed for seamless integration with Optimistix while
    maintaining the native pycutest-compatible representation in sif2jax.
    """

    problem: AbstractConstrainedMinimisation

    @property
    def y0_iD(self) -> int:
        return self.problem.y0_iD

    @property
    def provided_y0s(self) -> frozenset:
        return self.problem.provided_y0s

    @property
    def name(self) -> str:
        return self.problem.name

    def objective(self, y: PyTree[ArrayLike], args: PyTree[Any]) -> Scalar:
        return self.problem.objective(y, args)

    def y0(self) -> PyTree[ArrayLike]:
        return self.problem.y0()

    def args(self) -> PyTree[Any]:
        return self.problem.args()

    def expected_result(self) -> PyTree[ArrayLike]:
        return self.problem.expected_result()

    def expected_objective_value(self) -> Scalar | None:
        return self.problem.expected_objective_value()

    def bounds(self) -> PyTree[ArrayLike] | None:
        return self.problem.bounds()

    def constraint(self, y: PyTree[ArrayLike]) -> tuple[Array, Array]:
        """Convert constraints to Optimistix format: (h(y) = 0, g(y) ≥ 0)."""
        # Get constraint values and bounds
        eq_values, ineq_values = self.problem.constraint(y)
        eq_cl, ineq_cl = self.problem.constraint_bounds()[0]
        eq_cu, ineq_cu = self.problem.constraint_bounds()[1]

        # Equality constraints are already in the right format (should equal 0)
        if eq_values is not None:
            equalities, _ = jfu.ravel_pytree(eq_values)
        else:
            equalities = jnp.array([])

        # Process inequality constraints
        if ineq_values is None:
            inequalities = jnp.array([])
        else:
            # Flatten inequality values and bounds
            ineq_flat, _ = jfu.ravel_pytree(ineq_values)
            cl_flat, _ = jfu.ravel_pytree(ineq_cl)
            cu_flat, _ = jfu.ravel_pytree(ineq_cu)

            # Build inequality constraints using JAX operations
            # G-type: c(x) >= cl becomes c(x) - cl >= 0
            has_lower = jnp.isfinite(cl_flat)
            g_constraints = jnp.where(has_lower, ineq_flat - cl_flat, jnp.inf)

            # L-type: c(x) <= cu becomes -c(x) + cu >= 0
            has_upper = jnp.isfinite(cu_flat)
            l_constraints = jnp.where(has_upper, cu_flat - ineq_flat, jnp.inf)

            # Concatenate and filter out the inf values
            all_ineq = jnp.concatenate([g_constraints, l_constraints])
            inequalities = all_ineq[jnp.isfinite(all_ineq)]

        return equalities, inequalities

    def num_constraints(self) -> tuple[Int, Int, Int]:
        """Count constraints in Optimistix format."""
        # Get bounds
        eq_cl, ineq_cl = self.problem.constraint_bounds()[0]
        eq_cu, ineq_cu = self.problem.constraint_bounds()[1]

        # Count equalities (from the equality part of constraint())
        if eq_cl is None:
            num_equalities = 0
        else:
            eq_flat, _ = jfu.ravel_pytree(eq_cl)
            num_equalities = eq_flat.size

        # Count inequalities (may be more than original due to range splitting)
        if ineq_cl is None:
            num_inequalities = 0
        else:
            cl_flat, _ = jfu.ravel_pytree(ineq_cl)
            cu_flat, _ = jfu.ravel_pytree(ineq_cu)
            has_lower = jnp.isfinite(cl_flat)
            has_upper = jnp.isfinite(cu_flat)
            num_inequalities = jnp.sum(has_lower) + jnp.sum(has_upper)

        # Get number of bounds from the wrapped problem
        bounds = self.problem.bounds()
        if bounds is None:
            num_bounds = 0
        else:
            lower, upper = bounds
            lower_flat, _ = jfu.ravel_pytree(lower)
            upper_flat, _ = jfu.ravel_pytree(upper)
            num_bounds = jnp.sum(jnp.isfinite(lower_flat)) + jnp.sum(
                jnp.isfinite(upper_flat)
            )

        return num_equalities, num_inequalities, num_bounds

    def constraint_bounds(self) -> tuple[PyTree[ArrayLike], PyTree[ArrayLike]]:
        """Pass through constraint bounds from wrapped problem."""
        return self.problem.constraint_bounds()
