import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LUKSAN15(AbstractNonlinearEquations):
    """Problem 15 (sparse signomial) from Luksan.

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
        Y = jnp.array([35.8, 11.2, 6.2, 4.4], dtype=jnp.float64)

        # Create indices for vectorized computation
        # For each block j in range(s), variables are at i = 2*j, so:
        # x1 = x[2*j], x2 = x[2*j+1], x3 = x[2*j+2], x4 = x[2*j+3]
        j_indices = jnp.arange(s)
        i_indices = 2 * j_indices

        # Extract variables for all blocks
        x1 = x[i_indices]  # x[i] for all blocks
        x2 = x[i_indices + 1]  # x[i+1] for all blocks
        x3 = x[i_indices + 2]  # x[i+2] for all blocks
        x4 = x[i_indices + 3]  # x[i+3] for all blocks

        # Compute signomial term for all blocks: x1 * x2^2 * x3^3 * x4^4
        signom_vals = x1 * (x2**2) * (x3**3) * (x4**4)  # shape: (s,)

        # Create arrays for p and l values
        p_vals = jnp.array([1, 2, 3], dtype=jnp.float64)  # p = 1, 2, 3
        l_vals = jnp.array([1, 2, 3, 4], dtype=jnp.float64)  # l = 1, 2, 3, 4

        # Create meshgrids for vectorized computation
        P, L, J = jnp.meshgrid(
            p_vals, l_vals, j_indices, indexing="ij"
        )  # shapes: (3, 4, s)

        # Compute p2ol and pli for all combinations
        P2OL = P**2 / L  # shape: (3, 4, s)
        PLI = 1.0 / (P * L)  # shape: (3, 4, s)

        # Expand signom_vals to match the shape
        signom_expanded = signom_vals[None, None, :]  # shape: (1, 1, s)

        # Apply sign based on whether signom_val is positive
        sign_p = jnp.where(signom_expanded > 0, 1.0, -1.0)
        p_val = signom_expanded * sign_p  # This gives |signom_val|, shape: (1, 1, s)

        # Compute F = p2ol * p_val^pli for all combinations
        F_vals = P2OL * (p_val**PLI)  # shape: (3, 4, s)

        # Sum over p (axis=0) to get equation sums for each (l, j)
        eq_sums = jnp.sum(F_vals, axis=0)  # shape: (4, s)

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
