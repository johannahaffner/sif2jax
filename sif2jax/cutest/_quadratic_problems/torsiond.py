import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class TORSIOND(AbstractBoundedMinimisation):
    """The quadratic elastic torsion problem.

    The problem comes from the obstacle problem on a square.

    The square is discretized into (px-1)(py-1) little squares. The heights of the
    considered surface above the corners of these little squares are the problem
    variables. There are px**2 of them.

    The dimension of the problem is specified by Q, which is half the number
    discretization points along one of the coordinate direction. Since the number of
    variables is P**2, it is given by 4Q**2

    This is a variant of the problem stated in the report quoted below. It corresponds
    to the problem as distributed in MINPACK-2.

    Source: problem (c=10, starting point Z = origin) in
    J. More' and G. Toraldo,
    "On the Solution of Large Quadratic-Programming Problems with Bound Constraints",
    SIAM J. on Optimization, vol 1(1), pp. 93-113, 1991.

    SIF input: Ph. Toint, Dec 1989.
    modified by Peihuang Chen, according to MINPACK-2, Apr 1992.

    classification QBR2-MY-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    q: int = 36  # Default value (pycutest uses 36, not 37 from SIF)
    c: float = 10.0  # Force constant

    @property
    def n(self):
        """Number of variables = P^2 where P = 2*Q."""
        p = 2 * self.q
        return p * p

    @property
    def p(self):
        """Grid size."""
        return 2 * self.q

    @property
    def h(self):
        """Grid spacing."""
        return 1.0 / (self.p - 1)

    @property
    def y0(self):
        """Initial guess - all zeros."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def _xy_to_index(self, i, j):
        """Convert (i,j) grid coordinates to linear index using column-major order."""
        return j * self.p + i

    def _index_to_xy(self, idx):
        """Convert linear index to (i,j) grid coordinates using column-major order."""
        return idx % self.p, idx // self.p

    def objective(self, y, args):
        """Quadratic objective function.

        The objective is the sum of squared differences between neighboring
        grid points, scaled by the force constant.
        """
        del args
        p = self.p
        h2 = self.h * self.h
        c0 = h2 * self.c

        # Reshape to grid using column-major (Fortran) order
        x = y.reshape((p, p), order="F")

        # Terms from GL groups (left differences)
        # Vectorized: compute all left differences at once
        diff_i_left = x[1:, 1:] - x[:-1, 1:]  # Shape: (p-1, p-1)
        diff_j_left = x[1:, 1:] - x[1:, :-1]  # Shape: (p-1, p-1)
        gl_terms = 0.25 * (diff_i_left**2 + diff_j_left**2)

        # Terms from GR groups (right differences)
        # Vectorized: compute all right differences at once
        diff_i_right = x[1:, :-1] - x[:-1, :-1]  # Shape: (p-1, p-1)
        diff_j_right = x[:-1, 1:] - x[:-1, :-1]  # Shape: (p-1, p-1)
        gr_terms = 0.25 * (diff_i_right**2 + diff_j_right**2)

        # Linear terms from G groups
        # Vectorized: apply to interior points
        linear_terms = -c0 * x[1:-1, 1:-1]  # Shape: (p-2, p-2)

        # Sum all contributions
        obj = jnp.sum(gl_terms) + jnp.sum(gr_terms) + jnp.sum(linear_terms)

        return jnp.array(obj)

    @property
    def bounds(self):
        """Variable bounds based on distance to boundary."""
        p = self.p
        h = self.h

        # Initialize bounds as 2D grids
        lower_grid = jnp.full((p, p), -jnp.inf)
        upper_grid = jnp.full((p, p), jnp.inf)

        # Boundary variables are fixed at 0
        # Set all edges to 0
        lower_grid = lower_grid.at[0, :].set(0.0)  # Bottom edge
        upper_grid = upper_grid.at[0, :].set(0.0)
        lower_grid = lower_grid.at[p - 1, :].set(0.0)  # Top edge
        upper_grid = upper_grid.at[p - 1, :].set(0.0)
        lower_grid = lower_grid.at[:, 0].set(0.0)  # Left edge
        upper_grid = upper_grid.at[:, 0].set(0.0)
        lower_grid = lower_grid.at[:, p - 1].set(0.0)  # Right edge
        upper_grid = upper_grid.at[:, p - 1].set(0.0)

        # Create coordinate grids for vectorized distance computation
        i_grid, j_grid = jnp.meshgrid(
            jnp.arange(p, dtype=jnp.float64),
            jnp.arange(p, dtype=jnp.float64),
            indexing="ij",
        )

        # Compute distance to nearest boundary for each point
        dist_to_left = j_grid
        dist_to_right = float(p - 1) - j_grid
        dist_to_bottom = i_grid
        dist_to_top = float(p - 1) - i_grid

        # Minimum distance to any boundary
        min_dist = jnp.minimum(
            jnp.minimum(dist_to_left, dist_to_right),
            jnp.minimum(dist_to_bottom, dist_to_top),
        )

        # Scale by h
        dist_scaled = min_dist * h

        # Apply bounds based on distance, but preserve boundary zeros
        # Create mask for interior points
        interior_mask = (
            (i_grid > 0) & (i_grid < p - 1) & (j_grid > 0) & (j_grid < p - 1)
        )

        # Apply distance-based bounds only to interior points
        lower_grid = jnp.where(interior_mask, -dist_scaled, lower_grid)
        upper_grid = jnp.where(interior_mask, dist_scaled, upper_grid)

        # Flatten back to 1D using column-major order
        lower = lower_grid.flatten(order="F")
        upper = upper_grid.flatten(order="F")

        return lower, upper

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value for Q=37."""
        # From SIF file comments
        return jnp.array(-1.204200)
