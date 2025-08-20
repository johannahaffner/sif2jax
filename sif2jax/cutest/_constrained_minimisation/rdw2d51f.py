"""RDW2D51F problem implementation.

TODO: Human review needed
Attempts made:
1. Initial vectorized implementation with 5-point stencil finite difference
2. Fixed JAX compilation issues and dtype promotion errors
3. Corrected constraint method signature
4. Tried sign flip for constraint residual
Suspected issues:
- Finite difference discretization doesn't match complex finite element in SIF
- Need exact implementation of element contributions from A,B,C,D,P,Q,R,S elements
- Complex GROUP USES structure with multiple coefficients
Resources needed:
- Detailed finite element analysis of SIF structure
- Implementation of exact element matrix assembly
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class RDW2D51F(AbstractConstrainedMinimisation):
    r"""RDW2D51F problem.

    A finite-element approximation to the distributed optimal control problem

        min 1/2||u-v||_L2^2 + beta ||f||_L2^2

    subject to - nabla^2 u = f

    where v is given on and within the boundary of a unit [0,1] box in
    2 dimensions, and u = v on its boundary. The discretization uses
    quadrilateral elements. There are simple bounds on the controls f.

    The problem is stated as a quadratic program.

    Source: example 5.1 in
    T. Rees, H. S. Dollar and A. J. Wathen
    "Optimal solvers for PDE-constrained optimization"
    SIAM J. Sci. Comp. (to appear) 2009

    with the control bounds as specified in
    M. Stoll and A. J. Wathen
    "Preconditioning for PDE constrained optimization with
     control constraints"
    OUCL Technical Report 2009

    SIF input: Nick Gould, May 2009
              correction by S. Gratton & Ph. Toint, May 2024

    classification QLR2-AN-V-V
    """

    # Grid size parameter (power of 2)
    n: int = 256
    # Regularization parameter
    beta: float = 0.01

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n_var(self) -> int:
        """Number of variables: F(I,J) and U(I,J) for I,J = 0..n."""
        return 2 * (self.n + 1) ** 2

    @property
    def n_con(self) -> int:
        """Number of constraints: L(I,J) for I,J = 1..n-1."""
        return (self.n - 1) ** 2

    @property
    def bounds(self):
        """Bounds on variables."""
        # Grid spacing
        h = 1.0 / self.n
        n_points = self.n + 1

        # Create coordinate grids
        i_coords = jnp.arange(n_points)[:, None]
        j_coords = jnp.arange(n_points)[None, :]

        # Boundary mask
        boundary_mask = (
            (i_coords == 0)
            | (i_coords == self.n)
            | (j_coords == 0)
            | (j_coords == self.n)
        )

        # Control variable F bounds
        f_lower = jnp.zeros((n_points, n_points))
        f_upper = jnp.zeros((n_points, n_points))

        # Interior F bounds (position-dependent)
        x1 = i_coords.astype(jnp.float64) * h
        x2 = j_coords.astype(jnp.float64) * h
        arg = -(x1**2) - x2**2
        earg = jnp.exp(arg)
        ua = 0.1 * (2.0 - x1) * earg

        # Upper bounds based on position
        f_upper_vals = jnp.where(j_coords <= self.n // 2, 0.6, 0.9)

        # Apply bounds only to interior points
        f_lower = jnp.where(boundary_mask, 0.0, ua)
        f_upper = jnp.where(boundary_mask, 0.0, f_upper_vals)

        # State variable U bounds (boundary = target values)
        u_lower = jnp.zeros((n_points, n_points))
        u_upper = jnp.full((n_points, n_points), jnp.inf)

        # Compute target values V for boundary
        target_mask = (i_coords <= self.n // 2) & (j_coords <= self.n // 2)
        ri = 2.0 * i_coords.astype(jnp.float64) * h
        rj = 2.0 * j_coords.astype(jnp.float64) * h
        v_vals = jnp.where(target_mask, (ri - 1.0) ** 2 * (rj - 1.0) ** 2, 0.0)

        # Set boundary constraints
        u_lower = jnp.where(boundary_mask, v_vals, -jnp.inf)
        u_upper = jnp.where(boundary_mask, v_vals, jnp.inf)

        # Flatten and combine into arrays
        all_lower = jnp.concatenate([f_lower.flatten(), u_lower.flatten()])
        all_upper = jnp.concatenate([f_upper.flatten(), u_upper.flatten()])

        return (all_lower, all_upper)

    @property
    def y0(self):
        """Starting point."""
        # Grid size
        n_points = self.n + 1

        # Vectorized starting point computation
        # Note: coordinates not needed for zero initialization

        # F variables: all zeros
        f_vars = jnp.zeros((n_points, n_points))

        # U variables: all zeros (the RI+J/N line was commented out with *)
        u_vars = jnp.zeros((n_points, n_points))

        # Flatten and concatenate: F variables first, then U variables
        return jnp.concatenate([f_vars.flatten(), u_vars.flatten()])

    @property
    def args(self):
        """Arguments for the problem."""
        return None

    def _split_variables(self, x):
        """Split variables into F and U arrays."""
        n_points = self.n + 1
        n_grid = n_points**2

        f_vars = x[:n_grid].reshape((n_points, n_points))
        u_vars = x[n_grid:].reshape((n_points, n_points))

        return f_vars, u_vars

    def objective(self, y, args):
        """Objective function: 1/2||u-v||_L2^2 + beta ||f||_L2^2."""
        x = y
        f_vars, u_vars = self._split_variables(x)

        # Grid parameters
        h = 1.0 / self.n
        h_sq_36 = h**2 / 36.0
        beta_h_sq_36 = 2.0 * self.beta * h_sq_36
        n_points = self.n + 1

        # Compute target values V vectorized
        i_coords = jnp.arange(n_points)[:, None]
        j_coords = jnp.arange(n_points)[None, :]
        target_mask = (i_coords <= self.n // 2) & (j_coords <= self.n // 2)
        ri = 2.0 * i_coords.astype(jnp.float64) * h
        rj = 2.0 * j_coords.astype(jnp.float64) * h
        v_vals = jnp.where(target_mask, (ri - 1.0) ** 2 * (rj - 1.0) ** 2, 0.0)

        # Vectorized element extraction
        # Extract all element corners at once
        u_corners = jnp.stack(
            [
                u_vars[:-1, :-1],  # u1
                u_vars[:-1, 1:],  # u2
                u_vars[1:, :-1],  # u3
                u_vars[1:, 1:],  # u4
            ],
            axis=-1,
        )

        v_corners = jnp.stack(
            [
                v_vals[:-1, :-1],  # v1
                v_vals[:-1, 1:],  # v2
                v_vals[1:, :-1],  # v3
                v_vals[1:, 1:],  # v4
            ],
            axis=-1,
        )

        f_corners = jnp.stack(
            [
                f_vars[:-1, :-1],  # f1
                f_vars[:-1, 1:],  # f2
                f_vars[1:, :-1],  # f3
                f_vars[1:, 1:],  # f4
            ],
            axis=-1,
        )

        # Compute differences
        uv_diff = u_corners - v_corners

        # Element matrix M applied vectorized
        # M has pattern: diag(2,2,2,2) + cross terms
        def apply_element_matrix(vec):
            """Apply the 4x4 element mass matrix to corner values."""
            v1, v2, v3, v4 = vec[..., 0], vec[..., 1], vec[..., 2], vec[..., 3]
            return (
                2.0 * v1**2
                + 2.0 * v2**2
                + 2.0 * v3**2
                + 2.0 * v4**2
                + 2.0 * v1 * v2
                + 2.0 * v1 * v3
                + v1 * v4
                + v2 * v3
                + 2.0 * v2 * v4
                + 2.0 * v3 * v4
            )

        # Compute element contributions
        u_contrib = apply_element_matrix(uv_diff)
        f_contrib = apply_element_matrix(f_corners)

        # Sum all element contributions
        obj = jnp.sum(h_sq_36 * u_contrib + beta_h_sq_36 * f_contrib)

        return obj

    def constraint(self, y):
        """Constraint function: PDE residual -nabla^2 u = f."""
        x = y
        f_vars, u_vars = self._split_variables(x)

        # Grid parameters
        h = 1.0 / self.n

        # Fully vectorized 5-point stencil for interior points
        # Extract interior grid (excluding boundaries)
        u_interior = u_vars[1:-1, 1:-1]  # (n-1) x (n-1)
        u_left = u_vars[1:-1, :-2]  # left neighbors
        u_right = u_vars[1:-1, 2:]  # right neighbors
        u_up = u_vars[:-2, 1:-1]  # up neighbors
        u_down = u_vars[2:, 1:-1]  # down neighbors

        # Vectorized discrete Laplacian using 5-point stencil
        # -nabla^2 u ≈ (4*u - u_left - u_right - u_up - u_down) / h^2
        laplacian = (4.0 * u_interior - u_left - u_right - u_up - u_down) / h**2

        # Source terms at interior points
        f_interior = f_vars[1:-1, 1:-1]

        # PDE residual: nabla^2 u + f = 0 (equality constraints)
        equalities = laplacian + f_interior

        # Return (equalities, inequalities) - no inequality constraints
        return equalities.flatten(), None

    @property
    def expected_result(self):
        """Expected result - not provided in SIF."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value - not provided in SIF."""
        return None
