import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class DRCAVTY3(AbstractNonlinearEquations):
    """Driven cavity problem variant 3.

    This system of nonlinear equations models the stream function corresponding
    to an incompressible fluid flow in a driven cavity (after elimination of
    the vorticity).

    The problem is nonconvex.
    It differs from the problems DRCAVTY1 and DRCAVTY2 by the value
    chosen for the Reynolds number.

    Source: P.N. Brown and Y. Saad,
    "Hybrid Krylov Methods for Nonlinear Systems of Equations",
    SIAM J. Sci. Stat. Comput. 11, pp. 450-481, 1990.
    The boundary conditions have been set according to
    I.E. Kaporin and O. Axelsson,
    "On a class of nonlinear equation solvers based on the residual norm
    reduction over a sequence of affine subspaces",
    SIAM J, Sci. Comput. 16(1), 1995.

    SIF input: Ph. Toint, Jan 1995.
    Classification: NQR2-MY-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters from SIF file
    M: int = 63  # Mesh parameter (M=63 gives n=4489)
    RE: float = 5000.0  # Reynolds number (different from DRCAVTY1/2)

    def _get_problem_dimensions(self):
        """Calculate problem dimensions based on mesh parameter M.

        From pycutest analysis: n = (M+4)^2 = 67^2 = 4489 for M=63
        All variables Y(I,J) for I,J in [-1, M+2] are included.
        """
        return (self.M + 4) * (self.M + 4)  # Full grid: (M+4) x (M+4)

    def constraint(self, y):
        """Compute the constraint residuals F(x) = 0.

        Returns the residual vector for the nonlinear system.
        Interior points: equations E(I,J) for I,J in [1, M]
        Total equations: M × M = 63 × 63 = 3969
        """

        M = self.M
        RE = self.RE
        re_over_4 = RE / 4.0

        # y represents all variables Y(I,J) for I,J in [-1,M+2]
        # Reshape to full (M+4) x (M+4) grid
        Y = y.reshape(M + 4, M + 4)

        # Vectorized computation for all interior points
        # Interior points correspond to Y[2:M+2, 2:M+2]
        # (indices ii,jj where ii,jj in [2,M+1])
        # The equations E(I,J) are for I,J in [1,M]
        # In array terms where Y has shape (M+4, M+4) with indices for I,J in [-1,M+2]:
        # I,J = 1 corresponds to array indices [2, 2] (since -1->0, 0->1, 1->2)
        interior_slice = slice(
            2, M + 2
        )  # Corrected: equations for I,J in [1,M] -> array indices [2:M+2]

        # Extract all stencil values for linear part computation
        # Center point: Y[I,J] for I,J in [1,M]
        center = Y[interior_slice, interior_slice]

        # 4-connected neighbors (for I,J in [1,M] -> array indices [2:M+2])
        left = Y[interior_slice, slice(1, M + 1)]  # Y[I, J-1] -> array[2:M+2, 1:M+1]
        right = Y[interior_slice, slice(3, M + 3)]  # Y[I, J+1] -> array[2:M+2, 3:M+3]
        up = Y[slice(1, M + 1), interior_slice]  # Y[I-1, J] -> array[1:M+1, 2:M+2]
        down = Y[slice(3, M + 3), interior_slice]  # Y[I+1, J] -> array[3:M+3, 2:M+2]

        # 8-connected diagonal neighbors
        up_left = Y[
            slice(1, M + 1), slice(1, M + 1)
        ]  # Y[I-1, J-1] -> array[1:M+1, 1:M+1]
        up_right = Y[
            slice(1, M + 1), slice(3, M + 3)
        ]  # Y[I-1, J+1] -> array[1:M+1, 3:M+3]
        down_left = Y[
            slice(3, M + 3), slice(1, M + 1)
        ]  # Y[I+1, J-1] -> array[3:M+3, 1:M+1]
        down_right = Y[
            slice(3, M + 3), slice(3, M + 3)
        ]  # Y[I+1, J+1] -> array[3:M+3, 3:M+3]

        # Far neighbors for linear stencil - need to handle boundary carefully
        # For I,J in [1,M], the far neighbors are at I±2, J±2
        # In array indices: I,J -> I+1, J+1 (since array starts at 0 for I,J=-1)
        # So far neighbors are at (I+1)±2, (J+1)±2 = I-1,I+3 and J-1,J+3
        far_left = Y[
            interior_slice, slice(0, M)
        ]  # Y[I, J-2] -> indices (1:M+1, -1:M-1) -> (1:M+1, 0:M)
        far_right = Y[
            interior_slice, slice(4, M + 4)
        ]  # Y[I, J+2] -> indices (1:M+1, 3:M+3) -> (1:M+1, 4:M+4)
        far_up = Y[
            slice(0, M), interior_slice
        ]  # Y[I-2, J] -> indices (-1:M-1, 1:M+1) -> (0:M, 1:M+1)
        far_down = Y[
            slice(4, M + 4), interior_slice
        ]  # Y[I+2, J] -> indices (3:M+3, 1:M+1) -> (4:M+4, 1:M+1)

        # Linear part computation (vectorized) - from SIF file structure
        linear_part = (
            20.0 * center
            - 8.0 * up
            - 8.0 * down
            - 8.0 * left
            - 8.0 * right
            + 2.0 * up_right
            + 2.0 * down_left
            + 2.0 * up_left
            + 2.0 * down_right
            + far_up
            + far_down
            + far_left
            + far_right
        )

        # Nonlinear terms - X(I,J) computation (vectorized)
        x_aa = right - left  # Y[I, J+1] - Y[I, J-1]
        x_bb = (
            far_up
            + up_left
            + up_right
            - 4.0 * up
            + 4.0 * down
            - down_left
            - down_right
            - far_down
        )
        x_val = x_aa * x_bb

        # Z(I,J) computation (vectorized)
        z_aa = down - up  # Y[I+1, J] - Y[I-1, J]
        z_bb = (
            far_left
            + up_left
            + down_left
            - 4.0 * left
            + 4.0 * right
            - up_right
            - down_right
            - far_right
        )
        z_val = z_aa * z_bb

        # Final constraint residuals (vectorized)
        constraint_residuals = linear_part + re_over_4 * x_val - re_over_4 * z_val

        # Flatten to return as 1D constraint vector
        return (
            constraint_residuals.flatten(),
            None,
        )  # (equality_constraints, inequality_constraints)

    @property
    def y0(self):
        """Initial point - all zeros (from SIF bounds, FR means free variables)."""
        n = self._get_problem_dimensions()
        return jnp.zeros(n)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Not specified in SIF file
        n = self._get_problem_dimensions()
        return jnp.zeros(n)

    @property
    def bounds(self):
        """Return variable bounds based on SIF file structure.

        From SIF file:
        - FR 'DEFAULT': Free variables (most variables)
        - XX variables: Fixed at 0.0
        - ZX variables: Fixed at -H/2 or H/2 where H = 1/(M+2)
        """
        n = self._get_problem_dimensions()
        M = self.M
        H = 1.0 / (M + 2)  # Grid spacing, same as in constraint method

        # Initialize all as free variables
        lower_bounds = jnp.full(n, -jnp.inf)
        upper_bounds = jnp.full(n, jnp.inf)

        # Array layout: Y(I,J) for I,J in [-1, M+2] -> (M+4) x (M+4) array
        # I,J mapping: I=-1->0, I=0->1, I=1->2, ..., I=M+2->M+3

        # XX constraints: Fixed at 0.0
        # Bottom boundary: Y(-1,J) and Y(0,J) for J in [-1, M+2]
        for j in range(M + 4):  # J from -1 to M+2
            for i_idx in [0, 1]:  # I = -1,0 -> array indices 0,1
                idx = i_idx * (M + 4) + j
                lower_bounds = lower_bounds.at[idx].set(0.0)
                upper_bounds = upper_bounds.at[idx].set(0.0)

        # Left and right boundaries: Y(I,-1), Y(I,0), Y(I,M+1), Y(I,M+2) for I in [1,M]
        for i in range(2, M + 2):  # I from 1 to M -> array indices 2 to M+1
            for j_idx in [
                0,
                1,
                M + 2,
                M + 3,
            ]:  # J = -1,0,M+1,M+2 -> array indices 0,1,M+2,M+3
                idx = i * (M + 4) + j_idx
                lower_bounds = lower_bounds.at[idx].set(0.0)
                upper_bounds = upper_bounds.at[idx].set(0.0)

        # ZX constraints: Top boundary fixed at -H/2 or H/2
        # Y(M+1,J) fixed at -H/2, Y(M+2,J) fixed at H/2 for J in [-1, M+2]
        for j in range(M + 4):  # J from -1 to M+2
            # Y(M+1,J) -> array index (M+2, j) -> flattened index (M+2)*(M+4) + j
            idx_m1 = (M + 2) * (M + 4) + j
            lower_bounds = lower_bounds.at[idx_m1].set(-H / 2)
            upper_bounds = upper_bounds.at[idx_m1].set(-H / 2)

            # Y(M+2,J) -> array index (M+3, j) -> flattened index (M+3)*(M+4) + j
            idx_m2 = (M + 3) * (M + 4) + j
            lower_bounds = lower_bounds.at[idx_m2].set(H / 2)
            upper_bounds = upper_bounds.at[idx_m2].set(H / 2)

        return lower_bounds, upper_bounds

    @property
    def expected_objective_value(self):
        # Not applicable for nonlinear equations - return zero
        return jnp.array(0.0)
