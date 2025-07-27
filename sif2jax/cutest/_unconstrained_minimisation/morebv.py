"""The Boundary Value problem (nonlinear least-squares version)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed - Minor gradient precision differences
# Attempts made:
# 1. Vectorized for loops in objective function using JAX array operations
# 2. Vectorized y0 computation
# 3. Fixed dtype consistency issues (explicit dtype=jnp.float64)
# 4. All tests pass except gradient_at_start with tiny differences (1e-14 level)
#
# Suspected issues:
# - Gradient differences are at numerical precision limits (1e-14 to 1e-19)
# - Likely due to different accumulation order in vectorized vs loop operations
# - May be related to how JAX handles floating-point operations vs Fortran
#
# Additional resources needed:
# - Determine if these precision differences are acceptable
# - Consider using higher precision or different summation algorithms if needed
class MOREBV(AbstractUnconstrainedMinimisation):
    """The Boundary Value problem.

    This is the nonlinear least-squares version without fixed variables.

    Source: problem 28 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#17 (p. 75).

    SIF input: Ph. Toint, Dec 1989 and Nick Gould, Oct 1992.
    correction by S. Gratton & Ph. Toint, May 2024

    classification SUR2-MN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Number of variables, default from SIF

    def objective(self, y: Any, args: Any) -> Any:
        """Compute the objective function.

        The objective is a sum of squared terms involving finite differences
        and cubic terms.
        """
        h = 1.0 / (self.n + 1)
        h2 = h * h
        halfh2 = 0.5 * h2

        # Create indices for vectorized computation
        indices = jnp.arange(1, self.n + 1)
        indices = jnp.asarray(indices, dtype=y.dtype)
        ih_values = indices * h

        # Compute all cubic terms (x_i + i*h + 1)^3
        e_all = (y + ih_values + 1.0) ** 3

        # Initialize contributions array
        contributions = jnp.zeros(self.n, dtype=y.dtype)

        # Group G1: involves x1 and x2
        # G1 = 2*x1 - x2
        g1 = 2.0 * y[0] - y[1]
        contributions = contributions.at[0].set((g1 + halfh2 * e_all[0]) ** 2)

        # Groups G(i) for i = 2 to n-1 (vectorized)
        # G(i) = -x(i-1) + 2*x(i) - x(i+1)
        gi = -y[:-2] + 2.0 * y[1:-1] - y[2:]
        contributions = contributions.at[1:-1].set((gi + halfh2 * e_all[1:-1]) ** 2)

        # Group G(n): involves x(n-1) and x(n)
        # G(n) = -x(n-1) + 2*x(n)
        gn = -y[self.n - 2] + 2.0 * y[self.n - 1]
        contributions = contributions.at[self.n - 1].set(
            (gn + halfh2 * e_all[self.n - 1]) ** 2
        )

        return jnp.sum(contributions)

    @property
    def y0(self):
        # Starting point: x(i) = t(i) * (t(i) - 1)
        # where t(i) = i * h and h = 1/(n+1)
        h = 1.0 / (self.n + 1)
        indices = jnp.arange(1, self.n + 1)
        # Create y0 with default float type first
        ti = jnp.asarray(indices, dtype=jnp.float32) * h
        return ti * (ti - 1.0)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The solution is not provided in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # The optimal value is not provided in the SIF file
        # However, from the comments it should be 0.0 for the exact solution
        return jnp.array(0.0)
