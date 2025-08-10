import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class DIAGIQE(AbstractBoundedMinimisation):
    """A variable dimension indefinite quadratic problem with equispaced
    eigenvalues throughout the spectrum.

    lambda_i = i - n/2, i = 1, ... , n

    Source: simple test for GALAHAD gltr/glrt

    SIF input: Nick Gould, Feb 2019, corrected May 2024
    Classification: QBR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 1000

    def objective(self, y, args):
        n = y.shape[0]
        rn = n

        # Compute shift: -n/2
        shift = -rn / 2.0

        # Create eigenvalues: lambda_i = i + shift
        i_vals = jnp.arange(1, n + 1, dtype=y.dtype)
        h_vals = i_vals + shift

        # Compute x^T H x / 2 + g^T x
        # H is diagonal with h_vals on diagonal
        # g = ones vector
        hessian_term = 0.5 * jnp.sum(h_vals * y * y)
        gradient_term = jnp.sum(y)

        return hessian_term + gradient_term

    @property
    def y0(self):
        return jnp.ones(self.n)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The minimum occurs when Hx + g = 0, so x = -H^{-1}g
        n = self.n
        rn = n
        shift = -rn / 2.0
        i_vals = jnp.arange(1, n + 1, dtype=jnp.float64)
        h_vals = i_vals + shift
        return -1.0 / h_vals

    @property
    def expected_objective_value(self):
        x_opt = self.expected_result
        return self.objective(x_opt, self.args)

    @property
    def bounds(self):
        # From SIF file: bounds are [-100000, 1000000]
        lower = jnp.full(self.n, -100000.0)
        upper = jnp.full(self.n, 1000000.0)
        return lower, upper
