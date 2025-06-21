import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class ARGLINB(AbstractUnconstrainedMinimisation):
    """ARGLINB function.

    Variable dimension rank one linear problem.

    Source: Problem 33 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#93 (with different N and M)
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 200  # SIF file suggests 10, 50, 100, or 200
    m: int = 400  # SIF file suggests m >= n and values like 20, 100, 200, or 400

    def objective(self, y, args):
        del args
        n = self.n
        m = self.m

        # Based on AMPL model: each residual g_i = sum_j (i*j) * x_j - 1.0
        # This can be computed as a matrix-vector product
        # Create matrix A where A[i,j] = i*j (using 1-based indices)
        i_indices = jnp.arange(1, m + 1, dtype=jnp.float64)[:, None]  # Shape (m, 1)
        j_indices = jnp.arange(1, n + 1, dtype=jnp.float64)[None, :]  # Shape (1, n)
        A = i_indices * j_indices  # Shape (m, n)

        # Compute residuals as A @ y - 1
        residuals = A @ y - 1.0

        # Sum of squares of residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial value of 1.0 as specified in the SIF file
        return inexact_asarray(jnp.ones(self.n))

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        # The SIF file comments mention:
        # *LO SOLTN(10)          4.6341D+00
        # *LO SOLTN(50)          24.6268657
        # *LO SOLTN(100)         49.6259352
        # But no value for n=200 is provided
        return None
