import jax.numpy as jnp
import numpy as np

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class NONMSQRT(AbstractUnconstrainedMinimisation):
    """
    NONMSQRT problem.

    The "non-matrix square root problem" obtained from an error in
    writing a correct matrix square root problem B by Nocedal and Liu.

    Source:
    Ph. Toint

    SIF input: Ph. Toint, Dec 1989.

    classification SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    p: int = 70

    def objective(self, y, args):
        del args
        p = self.p

        # Reshape x from flat array to p x p matrix
        X = y.reshape(p, p)

        # Precompute constant matrix A = B @ B using numpy (no JAX tracing)
        k_vals = np.arange(1.0, p * p + 1.0).reshape(p, p)
        B = np.sin(k_vals * k_vals)
        B[2, 0] = 0.0
        A = B @ B

        # For each row i and position j:
        #   sum_k X[i,k]*X[i,j] = X[i,j] * row_sum[i]
        row_sums = jnp.sum(X, axis=1, keepdims=True)  # (p, 1)
        products = X * row_sums  # (p, p)

        # Squared Frobenius norm of residual
        diff = products - A
        return jnp.sum(diff**2)

    @property
    def y0(self):
        # Starting point based on SIF description
        p = self.p

        # Compute matrix B
        k_vals = jnp.arange(1.0, p * p + 1.0).reshape(p, p)
        B = jnp.sin(k_vals * k_vals)
        B = B.at[2, 0].set(0.0)

        # Starting point: X[i,j] = B[i,j] - 0.8 * sin(k^2)
        sk2 = jnp.sin(k_vals * k_vals)
        X_start = B - 0.8 * sk2

        # Flatten to 1D array
        return inexact_asarray(X_start.flatten())

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution not provided in detail in SIF file
        return None

    @property
    def expected_objective_value(self):
        # Solution values provided for some values of p
        p = self.p
        if p == 3:
            return jnp.array(0.075194572)
        elif p == 23:
            return jnp.array(61.327)
        else:
            # For p=70 and others, solution not provided
            return None

    def num_variables(self):
        return self.p * self.p
