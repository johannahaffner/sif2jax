from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed - Hessian computation issues
# Attempts made:
# 1. Vectorized nested loops using JAX meshgrid and broadcasting
# 2. Fixed matrix multiplication by using proper axis summation
# 3. Verified objective and gradient match the loop-based implementation
# 4. Confirmed vectorized implementation is mathematically correct
#
# Suspected issues:
# - Hessian shows differences up to 16 in specific matrix positions
# - Issue appears in column 8 (variable x9) interactions
# - Problem may be related to how JAX autodiff handles the vectorized operations
# - The combination of broadcasting, masking, and summation might interact
#   poorly with second-order automatic differentiation
#
# Additional resources needed:
# - Detailed comparison of Hessian values from pycutest vs JAX implementation
# - Investigation of whether the SIF file's use of logarithms for numerical
#   stability (ti^(j-2) computed as exp((j-2)*log(ti))) affects autodiff
# - Consider alternative vectorization strategies that might be more autodiff-friendly
class WATSON(AbstractUnconstrainedMinimisation):
    """Watson problem in 12 variables.

    This function is a nonlinear least squares with 31 groups. Each
    group has 1 nonlinear and 1 linear elements.

    Source: problem 20 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#128 (p. 100).

    SIF input: Ph. Toint, Dec 1989.
    (bug fix July 2007)

    classification SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 12  # Number of variables
    m: int = 31  # Number of groups

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function."""
        # For groups 1 to 29
        # Create array of t_i values: i/29 for i = 1 to 29
        i_vals = jnp.arange(1, 30, dtype=jnp.float64)
        ti_vals = i_vals / 29.0

        # Prepare indices for vectorized computation
        j_indices = jnp.arange(1, self.n + 1, dtype=jnp.float64)  # 1 to n

        # Create meshgrid for broadcasting: ti_vals (29,) and j_indices (n,)
        ti_grid, j_grid = jnp.meshgrid(ti_vals, j_indices, indexing="ij")

        # Linear part: for j from 2 to n, coefficient is (j-1) * ti^(j-2)
        # We compute for all j, then mask out j=1
        coefficients = (j_grid - 1) * (ti_grid ** (j_grid - 2))
        coefficients = jnp.where(j_grid > 1, coefficients, 0.0)

        # Linear sums: sum over j for each i
        linear_sums = jnp.sum(coefficients * y[None, :], axis=1)

        # Element MWSQ: weighted sum with ti^(j-1) weights
        weights = ti_grid ** (j_grid - 1)
        element_sums = jnp.sum(weights * y[None, :], axis=1)

        # MWSQ element function is -u^2
        element_vals = -element_sums * element_sums

        # Group contributions with L2 type: (linear + element - 1)^2
        group_vals = linear_sums + element_vals - 1.0
        obj = jnp.sum(group_vals * group_vals)

        # Group 30: x1 (constant is 0)
        group_30 = y[0] - 0.0
        obj = obj + group_30 * group_30

        # Group 31: x2 + MSQ(x1) - 1
        # MSQ element function is -x1^2
        element_31 = -y[0] * y[0]
        group_31 = y[1] + element_31 - 1.0
        obj = obj + group_31 * group_31

        return obj

    @property
    def y0(self):
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # From the SIF file comment: solution for n=12 is 2.27559922D-9
        # This suggests the optimal value is very close to zero
        return None  # Not provided explicitly

    @property
    def expected_objective_value(self):
        return jnp.array(2.27559922e-9, dtype=jnp.float64)
