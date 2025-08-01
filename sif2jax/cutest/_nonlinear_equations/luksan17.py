import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LUKSAN17(AbstractNonlinearEquations):
    """Problem 17 (sparse trigonometric) from Luksan.

    This is a system of nonlinear equations from the paper:
    L. Luksan
    "Hybrid methods in large sparse nonlinear least squares"
    J. Optimization Theory & Applications 89(3) 575-595 (1996)

    SIF input: Nick Gould, June 2017.

    classification NOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    s: int = 49  # Seed for dimensions (default from SIF)

    @property
    def n(self) -> int:
        """Number of variables: 2*S + 2."""
        return 2 * self.s + 2

    @property
    def m(self) -> int:
        """Number of equations: 4*S."""
        return 4 * self.s

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

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector."""
        del args  # Not used

        x = y
        s = self.s

        # Data values
        Y = jnp.array([30.6, 72.2, 124.4, 187.4], dtype=x.dtype)

        # Create indices for vectorized computation
        # For each block j in range(s), variables start at i = 2*j, so:
        # Variables needed: x[2*j], x[2*j+1], x[2*j+2], x[2*j+3] (for q=1,2,3,4)
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

        return residuals

    @property
    def expected_result(self) -> Array | None:
        """Expected optimal solution."""
        # No specific expected result provided in SIF
        return None

    @property
    def expected_objective_value(self) -> Array | None:
        """Expected objective value (sum of squares)."""
        # For nonlinear equations, the expected value is 0
        return jnp.array(0.0)

    def constraint(self, y: Array):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """No bounds for this problem."""
        return None
