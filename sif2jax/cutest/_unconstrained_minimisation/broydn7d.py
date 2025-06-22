import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class BROYDN7D(AbstractUnconstrainedMinimisation):
    """A seven diagonal variant of the Broyden tridiagonal system.

    Features a band far away from the diagonal.

    Source: Ph.L. Toint,
    "Some numerical results using a sparse matrix updating formula in
    unconstrained optimization",
    Mathematics of Computation, vol. 32(114), pp. 839-852, 1978.

    See also Buckley#84
    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Dimension of the problem (should be even)

    def objective(self, y, args):
        del args
        n = self.n
        half_n = n // 2

        # Compute g terms (tridiagonal structure)
        g_terms = jnp.zeros(n)

        # g₁ = -2x₂ + 1 + (3-2x₁)x₁
        g_terms = g_terms.at[0].set(-2.0 * y[1] + 1.0 + (3.0 - 2.0 * y[0]) * y[0])

        # gₙ = -xₙ₋₁ + 1 + (3-2xₙ)xₙ
        g_terms = g_terms.at[n - 1].set(
            -y[n - 2] + 1.0 + (3.0 - 2.0 * y[n - 1]) * y[n - 1]
        )

        # gᵢ = 1 - xᵢ₋₁ - 2xᵢ₊₁ + (3-2xᵢ)xᵢ for i = 2,...,N-1
        # (1-indexed becomes 1,...,N-2 in 0-indexed)
        if n > 2:
            middle_indices = jnp.arange(1, n - 1)
            g_terms = g_terms.at[middle_indices].set(
                1.0
                - y[middle_indices - 1]
                - 2.0 * y[middle_indices + 1]
                + (3.0 - 2.0 * y[middle_indices]) * y[middle_indices]
            )

        # Compute s terms (distant band)
        # sᵢ = xᵢ + x_{i+N/2} for i = 1,...,N/2
        # (1-indexed becomes 0,...,N/2-1 in 0-indexed)
        s_terms = y[:half_n] + y[half_n:]

        # Objective: sum of |gᵢ|^(7/3) + |sᵢ|^(7/3)
        g_objective = jnp.sum(jnp.abs(g_terms) ** (7.0 / 3.0))
        s_objective = jnp.sum(jnp.abs(s_terms) ** (7.0 / 3.0))

        return g_objective + s_objective

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return inexact_asarray(jnp.ones(self.n))

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to the SIF file comment (line 111),
        # the optimal objective value is 1.2701
        return jnp.array(1.2701)
