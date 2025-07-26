import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class TORSIOND(AbstractConstrainedQuadraticProblem):
    """The quadratic elastic torsion problem.

    The problem comes from the obstacle problem on a square.

    The square is discretized into (px-1)(py-1) little squares. The
    heights of the considered surface above the corners of these little
    squares are the problem variables. There are px**2 of them.

    The dimension of the problem is specified by Q, which is half the
    number discretization points along one of the coordinate
    direction. Since the number of variables is P**2, it is given by 4Q**2

    This is a variant of the problem stated in the report quoted below.
    It corresponds to the problem as distributed in MINPACK-2.

    Source: problem (c=10, starting point Z = origin) in
    J. More' and G. Toraldo,
    "On the Solution of Large Quadratic-Programming Problems with Bound
    Constraints",
    SIAM J. on Optimization, vol 1(1), pp. 93-113, 1991.

    SIF input: Ph. Toint, Dec 1989.
    modified by Peihuang Chen, according to MINPACK-2, Apr 1992.

    classification QBR2-MY-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    q: int = 37  # Default value from SIF file
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
        """Convert (i,j) grid coordinates to linear index."""
        return i * self.p + j

    def _index_to_xy(self, idx):
        """Convert linear index to (i,j) grid coordinates."""
        return idx // self.p, idx % self.p

    def objective(self, y, args):
        """Quadratic objective function.

        The objective is the sum of squared differences between neighboring
        grid points, scaled by the force constant.
        """
        del args
        p = self.p
        h2 = self.h * self.h
        c0 = h2 * self.c

        # Reshape to grid
        x = y.reshape((p, p))

        obj = 0.0

        # Terms from GL groups (left differences)
        for i in range(1, p):
            for j in range(1, p):
                diff_i = x[i, j] - x[i - 1, j]
                diff_j = x[i, j] - x[i, j - 1]
                obj += 0.25 * (diff_i**2 + diff_j**2)

        # Terms from GR groups (right differences)
        for i in range(p - 1):
            for j in range(p - 1):
                diff_i = x[i + 1, j] - x[i, j]
                diff_j = x[i, j + 1] - x[i, j]
                obj += 0.25 * (diff_i**2 + diff_j**2)

        # Linear terms from G groups
        for i in range(1, p - 1):
            for j in range(1, p - 1):
                obj -= c0 * x[i, j]

        return jnp.array(obj)

    @property
    def bounds(self):
        """Variable bounds based on distance to boundary."""
        p = self.p
        q = self.q
        h = self.h

        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)

        # Boundary variables are fixed at 0
        for j in range(p):
            # Bottom and top edges
            lower = lower.at[self._xy_to_index(0, j)].set(0.0)
            upper = upper.at[self._xy_to_index(0, j)].set(0.0)
            lower = lower.at[self._xy_to_index(p - 1, j)].set(0.0)
            upper = upper.at[self._xy_to_index(p - 1, j)].set(0.0)

        for i in range(1, p - 1):
            # Left and right edges
            lower = lower.at[self._xy_to_index(i, 0)].set(0.0)
            upper = upper.at[self._xy_to_index(i, 0)].set(0.0)
            lower = lower.at[self._xy_to_index(i, p - 1)].set(0.0)
            upper = upper.at[self._xy_to_index(i, p - 1)].set(0.0)

        # Interior bounds based on distance to boundary
        # Lower half of square
        for i in range(1, q + 1):
            for j in range(1, i + 1):
                dist = min(i - 1, j - 1) * h
                idx = self._xy_to_index(i, j)
                lower = lower.at[idx].set(-dist)
                upper = upper.at[idx].set(dist)

            for j in range(i + 1, p - i):
                dist = (i - 1) * h
                idx = self._xy_to_index(i, j)
                lower = lower.at[idx].set(-dist)
                upper = upper.at[idx].set(dist)

            for j in range(p - i, p - 1):
                dist = (p - j - 1) * h
                idx = self._xy_to_index(i, j)
                lower = lower.at[idx].set(-dist)
                upper = upper.at[idx].set(dist)

        # Upper half of square (symmetric)
        for i in range(q + 1, p - 1):
            for j in range(1, p - i):
                dist = min(p - i - 1, j - 1) * h
                idx = self._xy_to_index(i, j)
                lower = lower.at[idx].set(-dist)
                upper = upper.at[idx].set(dist)

            for j in range(p - i, i + 1):
                dist = (p - i - 1) * h
                idx = self._xy_to_index(i, j)
                lower = lower.at[idx].set(-dist)
                upper = upper.at[idx].set(dist)

            for j in range(i + 1, p - 1):
                dist = (p - j - 1) * h
                idx = self._xy_to_index(i, j)
                lower = lower.at[idx].set(-dist)
                upper = upper.at[idx].set(dist)

        return lower, upper

    def constraint(self, y):
        """No equality or inequality constraints beyond bounds."""
        return None, None

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value for Q=37."""
        # From SIF file comments
        return jnp.array(-1.204200)
