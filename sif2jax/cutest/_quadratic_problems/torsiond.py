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

    TODO: Human review needed
    Attempts made:
    1. Implemented vectorized objective function following SIF file structure
    2. Fixed dimension to q=36 (p=72, n=5184) to match pycutest
    3. Implemented distance-based bounds (not fixing boundaries at 0)
    4. Verified GL/GR groups compute forward/backward differences correctly
    5. Verified linear G groups apply -c0 coefficient to interior points

    Suspected issues:
    1. Objective mismatch at ones vector: our implementation gives -9.72 while
       pycutest gives 134.27. The difference is ~144 = 2*p which suggests a
       systematic difference in formulation.
    2. With all ones input, all differences are 0, so only linear terms contribute
       (negative). Pycutest's positive value suggests different computation.
    3. Grid spacing mismatch: pycutest bounds suggest h=1/73 while SIF file
       clearly specifies h=1/(p-1)=1/71 for p=72.
    4. All test cases pass except objective/gradient tests at non-zero vectors.

    Additional resources needed:
    1. Examine pycutest Fortran source to understand objective computation
    2. Verify if pycutest adds constant terms or uses different scaling
    3. Check if there's a transformation applied to test inputs
    4. Clarify the h calculation discrepancy
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
        """Variable bounds based on distance to boundary.

        Unlike what the SIF comments might suggest, the boundary variables
        are NOT fixed at 0 - they have bounds based on distance too.
        """
        p = self.p
        h = self.h

        # Create 2D coordinate grids
        i_grid, j_grid = jnp.meshgrid(jnp.arange(p), jnp.arange(p), indexing="ij")

        # Compute distance to each edge
        dist_to_bottom = i_grid  # Distance to i=0
        dist_to_top = p - 1 - i_grid  # Distance to i=p-1
        dist_to_left = j_grid  # Distance to j=0
        dist_to_right = p - 1 - j_grid  # Distance to j=p-1

        # Minimum distance to any edge
        min_dist = jnp.minimum(
            jnp.minimum(dist_to_bottom, dist_to_top),
            jnp.minimum(dist_to_left, dist_to_right),
        )

        # Scale by h
        dist_scaled = min_dist * h

        # Bounds are +/- the scaled distance
        lower_grid = -dist_scaled
        upper_grid = dist_scaled

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
