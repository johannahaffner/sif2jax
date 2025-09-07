import jax.numpy as jnp

from ..._problem import AbstractBoundedQuadraticProblem


class TORSIONF(AbstractBoundedQuadraticProblem):
    """The quadratic elastic torsion problem.

    The problem comes from the obstacle problem on a square.

    The square is discretized into (px-1)(py-1) little squares. The heights of the
    considered surface above the corners of these little squares are the problem
    variables. There are px**2 of them.

    The dimension of the problem is specified by Q, which is half the number
    discretization points along one of the coordinate direction. Since the number of
    variables is P**2, it is given by 4Q**2

    This is a variant of the problem stated in the report quoted below.
    It corresponds to the problem as distributed in MINPACK-2.

    Source: problem (c=20, starting point Z = origin) in
    J. More' and G. Toraldo,
    "On the Solution of Large Quadratic-Programming Problems with Bound Constraints",
    SIAM J. on Optimization, vol 1(1), pp. 93-113, 1991.

    SIF input: Ph. Toint, Dec 1989.
    modified by Peihuang Chen, according to MINPACK-2, Apr 1992.

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
        """Initial guess - all zeros (starting point Z = origin)."""
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
        """Quadratic objective function for TORSIONF.

        TORSIONF has the same GL/GR formulation as TORSIONA:
        - GR groups (I=1..P-1, J=1..P-1): A(I,J) + B(I,J) terms with 0.25 coefficient
        - GL groups (I=2..P, J=2..P): C(I,J) + D(I,J) terms with 0.25 coefficient
        - G groups (I=2..P-1, J=2..P-1): linear terms with -c0 coefficient

        Where:
        A(I,J): (X(I+1,J) - X(I,J))^2
        B(I,J): (X(I,J+1) - X(I,J))^2
        C(I,J): (X(I-1,J) - X(I,J))^2
        D(I,J): (X(I,J-1) - X(I,J))^2
        """
        del args
        p = self.p
        h2 = self.h * self.h
        c0 = h2 * self.c

        # Reshape to grid using column-major (Fortran) order
        x = y.reshape((p, p), order="F")

        # GR groups: range (I=1..P-1, J=1..P-1) - forward differences A,B
        # Vectorized implementation for A and B terms
        gr_region = x[:-1, :-1]  # Region [0:p-1, 0:p-1]

        # A terms: (X(I+1,J) - X(I,J))^2 for valid indices
        diff_a = x[1:, :-1] - gr_region  # x[i+1,j] - x[i,j]
        a_terms = 0.25 * jnp.sum(diff_a**2)

        # B terms: (X(I,J+1) - X(I,J))^2 for valid indices
        diff_b = x[:-1, 1:] - gr_region  # x[i,j+1] - x[i,j]
        b_terms = 0.25 * jnp.sum(diff_b**2)

        # GL groups: range (I=2..P, J=2..P) - backward differences C,D
        # Vectorized implementation for C and D terms
        gl_region = x[1:, 1:]  # Region [1:p, 1:p]

        # C terms: (X(I-1,J) - X(I,J))^2 for valid indices
        diff_c = x[:-1, 1:] - gl_region  # x[i-1,j] - x[i,j]
        c_terms = 0.25 * jnp.sum(diff_c**2)

        # D terms: (X(I,J-1) - X(I,J))^2 for valid indices
        diff_d = x[1:, :-1] - gl_region  # x[i,j-1] - x[i,j]
        d_terms = 0.25 * jnp.sum(diff_d**2)

        # G groups: range (I=2..P-1, J=2..P-1) - linear terms
        # Vectorized implementation for linear terms
        linear_terms = -c0 * jnp.sum(x[1:-1, 1:-1])

        obj = a_terms + b_terms + c_terms + d_terms + linear_terms
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
        return jnp.array(-2.850200)
