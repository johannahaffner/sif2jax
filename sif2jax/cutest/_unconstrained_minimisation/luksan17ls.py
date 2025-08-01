import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractUnconstrainedMinimisation


class LUKSAN17LS(AbstractUnconstrainedMinimisation):
    """Problem 17 (sparse trigonometric) from Luksan.

    This is a least squares problem from the paper:
    L. Luksan
    "Hybrid methods in large sparse nonlinear least squares"
    J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.

    least-squares version

    classification SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    s: int = 49  # Seed for dimensions (default from SIF)

    @property
    def n(self) -> int:
        """Number of variables: 2*S + 2."""
        return 2 * self.s + 2

    @property
    def y0(self) -> Array:
        """Initial guess: pattern (-0.8, 1.2, -1.2, 0.8) repeated."""
        pattern = jnp.array([-0.8, 1.2, -1.2, 0.8], dtype=jnp.float64)
        # Repeat pattern to cover all n variables
        full_pattern = jnp.tile(pattern, (self.n + 3) // 4)[: self.n]
        return full_pattern

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y: Array, args) -> Array:
        """Compute the least squares objective function.

        The objective is the sum of squares of M = 4*S residuals.
        Each block contributes 4 residuals corresponding to data values Y1-Y4.
        Each residual is the sum over q=1,2,3,4 of trigonometric terms minus the data.
        """
        del args  # Not used

        x = y
        s = self.s

        # Data values
        Y = jnp.array([30.6, 72.2, 124.4, 187.4], dtype=x.dtype)

        # Vectorized computation
        # Create indices for all blocks
        j_indices = jnp.arange(s)

        # Create arrays for l and q values
        l_vals = jnp.array([1, 2, 3, 4], dtype=x.dtype)  # l = 1, 2, 3, 4
        q_vals = jnp.array([1, 2, 3, 4], dtype=x.dtype)  # q = 1, 2, 3, 4

        # Create meshgrids for vectorized computation
        L, Q, J = jnp.meshgrid(
            l_vals, q_vals, j_indices, indexing="ij"
        )  # shapes: (4, 4, s)

        # Variable indices for each combination: var_idx = i + q - 1 = 2*j + q - 1
        var_indices = (2 * J.astype(x.dtype) + Q - 1).astype(
            jnp.int32
        )  # shape: (4, 4, s)

        # Extract variables for all combinations
        x_vals = x[var_indices]  # shape: (4, 4, s)

        # Compute coefficients
        # For sine term: a = -l * q^2
        a_sin = -L * (Q**2)  # shape: (4, 4, s)
        sin_terms = a_sin * jnp.sin(x_vals)  # shape: (4, 4, s)

        # For cosine term: a = l^2 * q
        a_cos = (L**2) * Q  # shape: (4, 4, s)
        cos_terms = a_cos * jnp.cos(x_vals)  # shape: (4, 4, s)

        # Total terms for each (l, q, j) combination
        total_terms = sin_terms + cos_terms  # shape: (4, 4, s)

        # Sum over q (axis=1) to get equation sums for each (l, j)
        eq_sums = jnp.sum(total_terms, axis=1)  # shape: (4, s)

        # Subtract Y values (broadcast Y to match shape)
        Y_expanded = Y[:, None]  # shape: (4, 1)
        residuals_matrix = eq_sums - Y_expanded  # shape: (4, s)

        # Flatten in the correct order: for each j, then for each l
        # The original order is: j=0,l=1; j=0,l=2; j=0,l=3; j=0,l=4; j=1,l=1; ...
        # residuals_matrix is (l, j), so we need to transpose and flatten
        residuals = residuals_matrix.T.flatten()  # shape: (4*s,)

        # Sum of squares (L2 group type in SIF)
        return jnp.sum(residuals**2)

    @property
    def expected_result(self) -> Array | None:
        """Expected optimal solution."""
        # No specific expected result provided in SIF
        return None

    @property
    def expected_objective_value(self) -> Array | None:
        """Expected objective value."""
        # For least squares, the expected value is 0
        return jnp.array(0.0)
