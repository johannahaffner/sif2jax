import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractUnconstrainedMinimisation


class LUKSAN16LS(AbstractUnconstrainedMinimisation):
    """Problem 16 (sparse exponential) from Luksan.

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
        Each residual is the sum over p=1,2,3 of exponential terms minus the data value.
        """
        del args  # Not used

        x = y
        s = self.s

        # Data values
        Y = jnp.array([35.8, 11.2, 6.2, 4.4], dtype=x.dtype)

        # Vectorized computation
        # Create indices for all blocks
        j_indices = jnp.arange(s)
        i_indices = 2 * j_indices  # Variable indices for each block

        # Extract variables for all blocks
        x1 = x[i_indices]  # x[i] for all blocks
        x2 = x[i_indices + 1]  # x[i+1] for all blocks
        x3 = x[i_indices + 2]  # x[i+2] for all blocks
        x4 = x[i_indices + 3]  # x[i+3] for all blocks

        # Compute S = x1 + 2*x2 + 3*x3 + 4*x4 for all blocks
        s_vals = x1 + 2.0 * x2 + 3.0 * x3 + 4.0 * x4  # shape: (s,)

        # Create arrays for p and l values
        p_vals = jnp.array([1, 2, 3], dtype=x.dtype)  # p = 1, 2, 3
        l_vals = jnp.array([1, 2, 3, 4], dtype=x.dtype)  # l = 1, 2, 3, 4

        # Create meshgrids for vectorized computation
        P, L, J = jnp.meshgrid(
            p_vals, l_vals, j_indices, indexing="ij"
        )  # shapes: (3, 4, s)

        # Compute p2ol and pli for all combinations
        P2OL = P**2 / L  # shape: (3, 4, s)
        PLI = 1.0 / (P * L)  # shape: (3, 4, s)

        # Expand s_vals to match the shape
        s_expanded = s_vals[None, None, :]  # shape: (1, 1, s)

        # Compute EXPARG = p2ol * exp(pli * s) for all combinations
        exparg_vals = P2OL * jnp.exp(PLI * s_expanded)  # shape: (3, 4, s)

        # Sum over p (axis=0) to get equation sums for each (l, j)
        eq_sums = jnp.sum(exparg_vals, axis=0)  # shape: (4, s)

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
