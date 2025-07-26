import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class TAME(AbstractConstrainedQuadraticProblem):
    """A simple constrained linear least-squares problem.

    Source:
    A.R. Conn, N. Gould and Ph.L. Toint,
    "The LANCELOT User's Manual",
    Dept of Maths, FUNDP, 1991.

    SIF input: Ph. Toint, Jan 1991.

    classification QLR2-AN-2-1
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 2

    @property
    def y0(self):
        """Initial guess - zeros."""
        return jnp.zeros(2)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function: (x - y)^2."""
        del args
        x, y_var = y[0], y[1]

        # The objective uses the SQUARE group type on (x - y)
        alpha = x - y_var
        return alpha**2

    @property
    def bounds(self):
        """No bounds on variables."""
        return None

    def constraint(self, y):
        """Linear equality constraint: x + y = 1."""
        x, y_var = y[0], y[1]

        # Equality constraint: x + y - 1 = 0
        eq = x + y_var - 1.0

        return jnp.array([eq]), None

    @property
    def expected_result(self):
        """Expected result: x = 0.5, y = 0.5."""
        return jnp.array([0.5, 0.5])

    @property
    def expected_objective_value(self):
        """Expected objective value at solution."""
        return jnp.array(0.0)
