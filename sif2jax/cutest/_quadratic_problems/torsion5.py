import jax.numpy as jnp

from ..._problem import AbstractBoundedQuadraticProblem


class TORSION5(AbstractBoundedQuadraticProblem):
    """The quadratic elastic torsion problem.

    The problem comes from the obstacle problem on a square.

    The square is discretized into (px-1)(py-1) little squares. The heights of the
    considered surface above the corners of these little squares are the problem
    variables. There are px**2 of them.

    The dimension of the problem is specified by Q, which is half the number
    discretization points along one of the coordinate direction. Since the number of
    variables is P**2, it is given by 4Q**2

    Source: problem (c=20, starting point U = upper bound) in
    J. More' and G. Toraldo,
    "On the Solution of Large Quadratic-Programming Problems with Bound Constraints",
    SIAM J. on Optimization, vol 1(1), pp. 93-113, 1991.

    SIF input: Ph. Toint, Dec 1989.

    classification QBR2-MY-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    q: int = 37  # Default value from SIF file (Q = 37, n = 5476)
    c: float = 20.0  # Force constant

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
        """Initial guess - start from upper bounds (starting point U)."""
        p = self.p
        h = self.h

        # Create coordinate grids for all points
        i_grid, j_grid = jnp.meshgrid(
            jnp.arange(p, dtype=jnp.float64),
            jnp.arange(p, dtype=jnp.float64),
            indexing="ij",
        )

        # Distance to boundary for all points
        dist_to_boundary = jnp.minimum(
            jnp.minimum(i_grid, p - 1 - i_grid), jnp.minimum(j_grid, p - 1 - j_grid)
        )

        # Scale by h
        upper_bounds = dist_to_boundary * h

        # Boundary points are 0, interior points use upper bounds
        on_boundary = (
            (i_grid == 0) | (i_grid == p - 1) | (j_grid == 0) | (j_grid == p - 1)
        )
        x = jnp.where(on_boundary, 0.0, upper_bounds)

        # Flatten to 1D using column-major order
        return x.flatten(order="F")

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

        From SIF file: each interior point (i,j) has 4 elements:
        A(I,J): 0.25 * (X(I+1,J) - X(I,J))^2
        B(I,J): 0.25 * (X(I,J+1) - X(I,J))^2
        C(I,J): 0.25 * (X(I-1,J) - X(I,J))^2
        D(I,J): 0.25 * (X(I,J-1) - X(I,J))^2
        Plus linear term: -c0 * X(I,J)
        """
        del args
        p = self.p
        h2 = self.h * self.h
        c0 = h2 * self.c

        # Reshape to grid using column-major (Fortran) order
        x = y.reshape((p, p), order="F")

        # For interior points [1:-1, 1:-1], compute 4 neighbor differences
        interior = x[1:-1, 1:-1]  # Shape: (p-2, p-2)

        # A terms: difference with point below (I+1,J)
        diff_down = x[2:, 1:-1] - interior  # x[i+1,j] - x[i,j]
        a_terms = 0.25 * diff_down**2

        # B terms: difference with point to the right (I,J+1)
        diff_right = x[1:-1, 2:] - interior  # x[i,j+1] - x[i,j]
        b_terms = 0.25 * diff_right**2

        # C terms: difference with point above (I-1,J)
        diff_up = x[:-2, 1:-1] - interior  # x[i-1,j] - x[i,j]
        c_terms = 0.25 * diff_up**2

        # D terms: difference with point to the left (I,J-1)
        diff_left = x[1:-1, :-2] - interior  # x[i,j-1] - x[i,j]
        d_terms = 0.25 * diff_left**2

        # Linear terms from G groups (coefficient -c0)
        linear_terms = -c0 * interior

        # Sum all contributions
        obj = jnp.sum(a_terms + b_terms + c_terms + d_terms + linear_terms)

        return jnp.array(obj)

    @property
    def bounds(self):
        """Variable bounds based on distance to boundary.

        Boundary variables are fixed at 0. Interior variables have bounds
        based on their distance to the boundary.
        """
        p = self.p
        h = self.h

        # Create 2D coordinate grids
        i_grid, j_grid = jnp.meshgrid(
            jnp.arange(p, dtype=jnp.float64),
            jnp.arange(p, dtype=jnp.float64),
            indexing="ij",
        )

        # Check if point is on boundary (any edge)
        on_boundary = (
            (i_grid == 0) | (i_grid == p - 1) | (j_grid == 0) | (j_grid == p - 1)
        )

        # For boundary points, bounds are [0, 0] (fixed)
        # For interior points, compute distance to boundary
        dist_to_boundary = jnp.where(
            on_boundary,
            0.0,  # Boundary points have distance 0 (will be fixed at 0)
            jnp.minimum(
                jnp.minimum(i_grid, p - 1 - i_grid),
                jnp.minimum(j_grid, p - 1 - j_grid),
            ),
        )

        # Scale by h
        dist_scaled = dist_to_boundary * h

        # Bounds are +/- the scaled distance
        lower_grid = jnp.where(on_boundary, 0.0, -dist_scaled)
        upper_grid = jnp.where(on_boundary, 0.0, dist_scaled)

        # Flatten to 1D using column-major order
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
