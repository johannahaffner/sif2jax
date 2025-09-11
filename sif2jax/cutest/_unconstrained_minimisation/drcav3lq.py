import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class DRCAV3LQ(AbstractUnconstrainedMinimisation):
    """Driven cavity problem, least-squares formulation, Reynolds number = 5000.

    TODO: Human review needed
    Attempts made:
    1. Tried adding Fortran column-major ordering for reshape - made error worse

    Suspected issues:
    - Large gradient error (48000) at element 68 with repeating_0123 pattern
    - Element 68 corresponds to Y(1,1) in column-major ordering
    - May be related to boundary condition handling or stencil computation

    Resources needed:
    - Check pycutest Fortran source for exact stencil implementation
    - Verify boundary condition handling matches SIF file

    This system of nonlinear equations models the stream function corresponding
    to an incompressible fluid flow in a driven cavity (after elimination of
    the vorticity). The system is solved in the least-squares sense.

    The problem is nonconvex. It differs from DRCAV1LQ and DRCAV2LQ by the
    value chosen for the Reynolds number.

    Source: P.N. Brown and Y. Saad,
    "Hybrid Krylov Methods for Nonlinear Systems of Equations",
    SIAM J. Sci. Stat. Comput. 11, pp. 450-481, 1990.
    The boundary conditions have been set according to
    I.E. Kaporin and O. Axelsson,
    "On a class of nonlinear equation solvers based on the residual norm
    reduction over a sequence of affine subspaces",
    SIAM J, Sci. Comput. 16(1), 1995.

    SIF input: Ph. Toint, Jan 1995.
    Classification: OXR2-MY-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters from SIF file
    M: int = 63  # Mesh parameter (M=63 gives n=4489)
    RE: float = 5000.0  # Reynolds number (different from DRCAV1LQ and DRCAV2LQ)

    def _get_problem_dimensions(self):
        """Calculate problem dimensions based on mesh parameter M.

        From pycutest analysis: n = (M+4)^2 = 67^2 = 4489 for M=63
        All variables Y(I,J) for I,J in [-1, M+2] are included.
        """
        return (self.M + 4) * (self.M + 4)  # Full grid: (M+4) x (M+4)

    def objective(self, y, args):
        """Compute the least-squares objective function."""
        del args

        M = self.M
        RE = self.RE
        re_over_4 = RE / 4.0

        # y represents all variables Y(I,J) for I,J in [-1,M+2]
        # Reshape to full (M+4) x (M+4) grid
        Y = y.reshape(M + 4, M + 4)

        # Do not override boundary values - they are handled by the problem structure
        # The input y already contains the correct boundary values

        # Vectorized computation for all interior points
        # Interior points correspond to Y[2:M+2, 2:M+2]
        # (indices ii,jj where ii,jj in [2,M+1])
        interior_slice = slice(2, M + 2)

        # Extract all stencil values for linear part computation
        # Center point: Y[ii, jj]
        center = Y[interior_slice, interior_slice]

        # 4-connected neighbors
        left = Y[interior_slice, slice(1, M + 1)]  # Y[ii, jj-1]
        right = Y[interior_slice, slice(3, M + 3)]  # Y[ii, jj+1]
        up = Y[slice(1, M + 1), interior_slice]  # Y[ii-1, jj]
        down = Y[slice(3, M + 3), interior_slice]  # Y[ii+1, jj]

        # 8-connected diagonal neighbors
        up_left = Y[slice(1, M + 1), slice(1, M + 1)]  # Y[ii-1, jj-1]
        up_right = Y[slice(1, M + 1), slice(3, M + 3)]  # Y[ii-1, jj+1]
        down_left = Y[slice(3, M + 3), slice(1, M + 1)]  # Y[ii+1, jj-1]
        down_right = Y[slice(3, M + 3), slice(3, M + 3)]  # Y[ii+1, jj+1]

        # Far neighbors for linear stencil
        far_left = Y[interior_slice, slice(0, M)]  # Y[ii, jj-2]
        far_right = Y[interior_slice, slice(4, M + 4)]  # Y[ii, jj+2]
        far_up = Y[slice(0, M), interior_slice]  # Y[ii-2, jj]
        far_down = Y[slice(4, M + 4), interior_slice]  # Y[ii+2, jj]

        # Linear part computation (vectorized)
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

        # X(I,J) computation (vectorized)
        x_aa = right - left  # Y[ii, jj+1] - Y[ii, jj-1]
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
        z_aa = down - up  # Y[ii+1, jj] - Y[ii-1, jj]
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

        # Group residual computation (vectorized)
        group_val = linear_part + re_over_4 * x_val - re_over_4 * z_val
        residual_squared = group_val * group_val

        # Sum all residuals
        total_residual = jnp.sum(residual_squared)

        return total_residual

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

    def num_constraints(self):
        """Return number of constraints and finite bounds.

        This problem has 520 variables fixed to zero (boundary conditions).
        """
        # Same boundary structure as DRCAV1LQ
        num_fixed = 520
        num_finite_bounds = 2 * num_fixed
        num_inequality = 0
        num_equality = 0

        return num_inequality, num_equality, num_finite_bounds

    @property
    def expected_objective_value(self):
        # From SIF: LO DRCAV3LQ 0.0 (lower bound is 0)
        return jnp.array(0.0)
