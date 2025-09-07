"""OBSTCL series - Quadratic obstacle problems by Dembo and Tulowitzki."""

import jax.numpy as jnp

from sif2jax._problem import AbstractBoundedMinimisation


class OBSTCLBase(AbstractBoundedMinimisation):
    """Base class for OBSTCL series quadratic obstacle problems.

    The problem comes from the obstacle problem on a rectangle.
    The rectangle is discretized into (px-1)(py-1) little rectangles. The
    heights of the considered surface above the corners of these little
    rectangles are the problem variables. There are px*py of them.

    Source:
    R. Dembo and U. Tulowitzki,
    "On the minimization of quadratic functions subject to box
    constraints",
    WP 71, Yale University (New Haven, USA), 1983.

    See also More 1989 (various problems and starting points)

    SIF input: Ph. Toint, Dec 1989.
    correction by S. Gratton & Ph. Toint, May 2024

    classification QBR2-AY-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    px: int = 100
    py: int = 100
    c: float = 1.0

    @property
    def hx(self):
        return 1.0 / (self.px - 1)

    @property
    def hy(self):
        return 1.0 / (self.py - 1)

    @property
    def hxhy(self):
        return self.hx * self.hy

    @property
    def hy_4hx(self):
        return 0.25 * self.hy / self.hx

    @property
    def hx_4hy(self):
        return 0.25 * self.hx / self.hy

    @property
    def linc(self):
        return -self.hxhy * self.c

    @property
    def n(self) -> int:
        """Number of variables."""
        return self.px * self.py

    @property
    def args(self):
        return ()

    def reshape_to_grid(self, x):
        """Reshape flat vector to grid."""
        return x.reshape(self.py, self.px)

    def flatten_grid(self, grid):
        """Flatten grid to vector."""
        return grid.flatten()

    def objective(self, y, args):
        """Compute objective function - quadratic form from finite differences."""
        grid = self.reshape_to_grid(y)

        # Extract interior points (2:py-1, 2:px-1) in 0-indexed notation
        interior = grid[1:-1, 1:-1]

        # Neighbor grids
        north = grid[2:, 1:-1]
        south = grid[:-2, 1:-1]
        east = grid[1:-1, 2:]
        west = grid[1:-1, :-2]

        # Compute quadratic form using vectorized operations
        # Each interior node contributes: linc * x(i,j) + sum of squared differences
        linear_term = jnp.sum(interior) * self.linc

        # Squared differences weighted by grid spacing
        diff_north = jnp.square(north - interior)
        diff_south = jnp.square(south - interior)
        diff_east = jnp.square(east - interior)
        diff_west = jnp.square(west - interior)

        quadratic_term = self.hy_4hx * (
            jnp.sum(diff_north) + jnp.sum(diff_south)
        ) + self.hx_4hy * (jnp.sum(diff_east) + jnp.sum(diff_west))

        return linear_term + quadratic_term

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        # Not specified in SIF file
        return None

    def _compute_obstacle_a_vectorized(self):
        """Compute obstacle values for all interior points using vectorization.

        Returns:
            2D array of obstacle values for interior points
        """
        # Create meshgrid for interior points (1 to py-2, 1 to px-2)
        i_indices, j_indices = jnp.meshgrid(
            jnp.arange(1, self.py - 1), jnp.arange(1, self.px - 1), indexing="ij"
        )

        # The SIF file uses X(I,J) where I is row, J is column
        # But pycutest uses column-major ordering, so we swap
        xi1 = j_indices.astype(jnp.float64) * self.hy
        xi2 = i_indices.astype(jnp.float64) * self.hx

        # Problem A: sin(3.2 * xi1) * sin(3.3 * xi2)
        return jnp.sin(3.2 * xi1) * jnp.sin(3.3 * xi2)

    def _compute_obstacle_a(self, i_indices, j_indices):
        """Compute obstacle values for Problem A variant.

        Args:
            i_indices: 0-based row indices (for SIF's I coordinate)
            j_indices: 0-based column indices (for SIF's J coordinate)
        """
        # The SIF file uses X(I,J) where I is row, J is column
        # But when comparing with pycutest, the formula needs swapping
        # This suggests column-major ordering in the flattened array
        # So we swap: use j for the first coordinate, i for the second
        xi1 = jnp.asarray(j_indices, dtype=jnp.float64) * self.hy
        xi2 = jnp.asarray(i_indices, dtype=jnp.float64) * self.hx

        # Problem A: sin(3.2 * xi1) * sin(3.3 * xi2)
        # From SIF: RM 3XI1 XI1 3.2 and RM 3XI2 XI2 3.3
        return jnp.sin(3.2 * xi1) * jnp.sin(3.3 * xi2)

    def _compute_obstacle_b_vectorized(self):
        """Compute obstacle values for all interior points using vectorization.

        Returns:
            2D array of obstacle values for interior points
        """
        # Create meshgrid for interior points (1 to py-2, 1 to px-2)
        i_indices, j_indices = jnp.meshgrid(
            jnp.arange(1, self.py - 1), jnp.arange(1, self.px - 1), indexing="ij"
        )

        # The SIF file uses X(I,J) where I is row, J is column
        # But pycutest uses column-major ordering, so we swap
        xi1 = j_indices.astype(jnp.float64) * self.hy
        xi2 = i_indices.astype(jnp.float64) * self.hx

        # Problem B: sin(9.2 * xi1) * sin(9.3 * xi2)
        return jnp.sin(9.2 * xi1) * jnp.sin(9.3 * xi2)

    def _compute_obstacle_b(self, i_indices, j_indices):
        """Compute obstacle values for Problem B variant."""
        # Use same swapping as in Problem A
        xi1 = jnp.asarray(j_indices, dtype=jnp.float64) * self.hy
        xi2 = jnp.asarray(i_indices, dtype=jnp.float64) * self.hx

        # Problem B: sin(9.2 * xi1) * sin(9.3 * xi2)
        # From SIF: RM 3XI1 XI1 9.2 and RM 3XI2 XI2 9.3
        return jnp.sin(9.2 * xi1) * jnp.sin(9.3 * xi2)


class OBSTCLAE(OBSTCLBase):
    """OBSTCL Problem A with starting point E.

    Starting point E: All interior variables set to 1.0, boundary to 0.0
    """

    @property
    def y0(self):
        """Initial guess - starting point E."""
        grid = jnp.ones((self.py, self.px))

        # Set boundaries to 0
        grid = grid.at[0, :].set(0.0)  # Bottom edge
        grid = grid.at[-1, :].set(0.0)  # Top edge
        grid = grid.at[:, 0].set(0.0)  # Left edge
        grid = grid.at[:, -1].set(0.0)  # Right edge

        return self.flatten_grid(grid)

    @property
    def bounds(self):
        """Variable bounds."""
        # Initialize bounds
        lower = jnp.full((self.py, self.px), -jnp.inf)
        upper = jnp.full(
            (self.py, self.px), 2000.0
        )  # From SIF: XU OBSTCLAE 'DEFAULT' 2000.0

        # Compute obstacle values for all interior points at once
        obstacle_interior = self._compute_obstacle_a_vectorized()

        # Set lower bounds for interior points
        lower = lower.at[1:-1, 1:-1].set(obstacle_interior)

        # Boundaries fixed at 0
        # Top and bottom edges
        lower = lower.at[0, :].set(0.0)
        upper = upper.at[0, :].set(0.0)
        lower = lower.at[-1, :].set(0.0)
        upper = upper.at[-1, :].set(0.0)

        # Left and right edges
        lower = lower.at[:, 0].set(0.0)
        upper = upper.at[:, 0].set(0.0)
        lower = lower.at[:, -1].set(0.0)
        upper = upper.at[:, -1].set(0.0)

        return self.flatten_grid(lower), self.flatten_grid(upper)


class OBSTCLAL(OBSTCLBase):
    """OBSTCL Problem A with starting point L.

    Starting point L: Lower obstacle values
    """

    @property
    def y0(self):
        """Initial guess - starting point L (lower obstacle values)."""
        grid = jnp.zeros((self.py, self.px))

        # Compute obstacle values for all interior points at once
        obstacle_interior = self._compute_obstacle_a_vectorized()

        # Set interior points to lower obstacle values
        grid = grid.at[1:-1, 1:-1].set(obstacle_interior)

        # Boundaries remain at 0 (already set)
        return self.flatten_grid(grid)

    @property
    def bounds(self):
        """Variable bounds - same as OBSTCLAE."""
        return OBSTCLAE().bounds


class OBSTCLBL(OBSTCLBase):
    """OBSTCL Problem B with starting point L.

    Problem B: sin(9.2 * xi1) * sin(9.3 * xi2)
    Starting point L: Lower obstacle values
    """

    @property
    def y0(self):
        """Initial guess - starting point L (lower obstacle values)."""
        grid = jnp.zeros((self.py, self.px))

        # Compute obstacle values for all interior points at once
        obstacle_interior = self._compute_obstacle_b_vectorized()

        # Set interior points to lower obstacle values (L^3)
        grid = grid.at[1:-1, 1:-1].set(obstacle_interior**3)

        # Boundaries remain at 0 (already set)
        return self.flatten_grid(grid)

    @property
    def bounds(self):
        """Variable bounds for Problem B."""
        # Initialize bounds
        lower = jnp.full((self.py, self.px), -jnp.inf)
        upper = jnp.full(
            (self.py, self.px), 2000.0
        )  # From SIF: XU OBSTCLBL 'DEFAULT' 2000.0

        # Compute obstacle values for all interior points at once
        obstacle_interior = self._compute_obstacle_b_vectorized()

        # Set lower bounds for interior points to L^3
        lower = lower.at[1:-1, 1:-1].set(obstacle_interior**3)

        # Set upper bounds for interior points to L^2 + 0.02
        upper = upper.at[1:-1, 1:-1].set(obstacle_interior**2 + 0.02)

        # Boundaries fixed at 0
        # Top and bottom edges
        lower = lower.at[0, :].set(0.0)
        upper = upper.at[0, :].set(0.0)
        lower = lower.at[-1, :].set(0.0)
        upper = upper.at[-1, :].set(0.0)

        # Left and right edges
        lower = lower.at[:, 0].set(0.0)
        upper = upper.at[:, 0].set(0.0)
        lower = lower.at[:, -1].set(0.0)
        upper = upper.at[:, -1].set(0.0)

        return self.flatten_grid(lower), self.flatten_grid(upper)


class OBSTCLBM(OBSTCLBase):
    """OBSTCL Problem B with starting point M.

    Problem B: sin(9.2 * xi1) * sin(9.3 * xi2)
    Starting point M: Midpoint between lower and upper obstacle values
    """

    @property
    def y0(self):
        """Initial guess - starting point M (midpoint)."""
        grid = jnp.zeros((self.py, self.px))

        # Compute obstacle values for all interior points at once
        obstacle_interior = self._compute_obstacle_b_vectorized()

        # Set interior points to midpoint between lower and upper obstacles
        # Lower is L^3, Upper is L^2 + 0.02
        lower_obstacle = obstacle_interior**3
        upper_obstacle = obstacle_interior**2 + 0.02
        midpoint = 0.5 * (lower_obstacle + upper_obstacle)

        grid = grid.at[1:-1, 1:-1].set(midpoint)

        # Boundaries remain at 0 (already set)
        return self.flatten_grid(grid)

    @property
    def bounds(self):
        """Variable bounds - same as OBSTCLBL."""
        return OBSTCLBL().bounds


class OBSTCLBU(OBSTCLBase):
    """OBSTCL Problem B with starting point U.

    Problem B: sin(9.2 * xi1) * sin(9.3 * xi2)
    Starting point U: Upper obstacle values
    """

    @property
    def y0(self):
        """Initial guess - starting point U (upper obstacle values)."""
        grid = jnp.zeros((self.py, self.px))

        # Compute obstacle values for all interior points at once
        obstacle_interior = self._compute_obstacle_b_vectorized()

        # Set interior points to upper obstacle values
        # Upper is L^2 + 0.02
        upper_obstacle = obstacle_interior**2 + 0.02

        grid = grid.at[1:-1, 1:-1].set(upper_obstacle)

        # Boundaries remain at 0 (already set)
        return self.flatten_grid(grid)

    @property
    def bounds(self):
        """Variable bounds - same as OBSTCLBL."""
        return OBSTCLBL().bounds
