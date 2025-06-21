import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class TOINTGSS(AbstractUnconstrainedMinimisation):
    """TOINTGSS problem - Toint's Gaussian problem.

    This problem has N-2 trivial groups, all of which have 1 nonlinear element.

    Source: problem 21 in
    Ph.L. Toint,
    "Test problems for partially separable optimization and results
    for the routine PSPMIN",
    Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

    SIF input: Ph. Toint, Dec 1989, corrected Nick Gould, July 1993.

    Classification: OUR2-AY-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    _n: int = 5000

    @property
    def n(self):
        """Number of variables."""
        return self._n

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        x = y
        n = self.n

        # Parameter: 10 / (n-2)
        ap = 10.0 / (n - 2)

        # Extract all overlapping triplets using slicing
        # v1 = x[0:n-2], v2 = x[1:n-1], v3 = x[2:n]
        v1 = x[:-2]
        v2 = x[1:-1]
        v3 = x[2:]

        # Compute all elements vectorized
        u1 = v1 - v2
        u2 = v3

        alpha = 0.1
        u1sq = u1 * u1
        u2sq = u2 * u2
        t = alpha + u2sq
        au2sq = ap + u2sq

        expa = jnp.exp(-u1sq / t)
        tmexpa = 2.0 - expa

        # Element values
        elements = au2sq * tmexpa

        # Sum all elements
        return jnp.sum(elements)

    def y0(self):
        """Initial guess."""
        return jnp.full(self.n, 3.0)

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def expected_result(self):
        """Expected optimal solution."""
        return None  # Not provided in SIF

    def expected_objective_value(self):
        """Expected optimal objective value."""
        return None  # Not provided in SIF
