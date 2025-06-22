import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class INTEQNELS(AbstractUnconstrainedMinimisation):
    """The discrete integral equation function in least-squares form.

    This is problem 29 from:
    J.J. Moré, B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    The problem discretizes an integral equation. The function f_i(x) is defined as:
    f_i(x) = x_i + h[(1-t_i)∑_{j=0}^i t_j(x_j + t_j + 1)³
        + t_i ∑_{j=i+1}^{n-1} (1-t_j)(x_j + t_j + 1)³]/2

    where h = 1/(n-1), t_i = ih for i = 0, ..., n-1.
    The objective is the sum of squares: ∑_{i=0}^{n-1} f_i(x)².

    Initial point: x_i = t_i(t_i - 1) for i = 0, ..., n-1.

    SIF input: Ph. Toint, Feb 1990.
    Modification to remove fixed variables: Nick Gould, Oct 2015.
    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 502  # Default value to match pycutest

    def objective(self, y, args):
        """Compute the objective function value as sum of squares of f_i(x)."""
        n = self.n
        h = 1.0 / (n - 1)  # Match the starting point indexing

        # t_i = ih for i = 0, 1, ..., n-1 (to match starting point)
        t = inexact_asarray(jnp.arange(n)) * h

        # y represents x_0, x_1, ..., x_{n-1}
        x = y

        def compute_f_i(i):
            """Compute f_i(x) for the i-th equation (0-indexed)."""
            t_i = t[i]
            x_i = x[i]

            # Use masking for both sums to avoid dynamic arange
            all_j = jnp.arange(n)
            t_j = t[all_j]
            x_j = x[all_j]

            # First sum: ∑_{j=0}^i t_j(x_j + t_j + 1)³
            first_mask = all_j <= i
            first_values = jnp.where(first_mask, t_j * (x_j + t_j + 1.0) ** 3, 0.0)
            first_sum = jnp.sum(jnp.asarray(first_values))

            # Second sum: ∑_{j=i+1}^{n-1} (1-t_j)(x_j + t_j + 1)³
            second_mask = all_j > i
            second_values = jnp.where(
                second_mask, (1.0 - t_j) * (x_j + t_j + 1.0) ** 3, 0.0
            )
            second_sum = jnp.sum(jnp.asarray(second_values))

            # Compute f_i(x)
            integral_part = h * ((1.0 - t_i) * first_sum + t_i * second_sum) / 2.0
            f_i = x_i + integral_part

            return f_i

        # Compute all f_i values and sum their squares
        i_values = jnp.arange(n)
        f_values = jax.vmap(compute_f_i)(i_values)

        return jnp.sum(f_values**2)

    def y0(self):
        """Initial point: ξ_i = t_i(t_i - 1) where t_i = ih.

        From the reference: x_0 = (ξ_i) where ξ_i = t_j(t_j - 1).

        Based on PyCUTEst output, it seems to use i = 0, 1, ..., n-1 with h = 1/(n-1).
        This gives t_i = ih and ξ_i = t_i(t_i - 1) starting with ξ_0 = 0.
        """
        n = self.n
        h = jnp.array(1.0) / (n - 1)  # This matches PyCUTEst behavior
        # For i = 0, 1, 2, ..., n-1, compute t_i = i*h and ξ_i = t_i*(t_i - 1)
        i_vals = jnp.arange(n)
        t_vals = i_vals.astype(h.dtype) * h
        xi_vals = t_vals * (t_vals - 1.0)
        return inexact_asarray(xi_vals)

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The solution is not specified in the SIF file."""
        return None

    def expected_objective_value(self):
        """The solution value is not specified in the SIF file."""
        return None
