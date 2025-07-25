import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HIMMELBC(AbstractConstrainedMinimisation):
    """A 2 variables problem by Himmelblau.

    Source: problem 28 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.

    See Buckley#6 (p. 63)

    SIF input: Ph. Toint, Dec 1989.

    classification: NQR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # 2 variables
    m_eq: int = 2  # 2 equality constraints
    m_ineq: int = 0  # no inequality constraints

    @property
    def y0(self):
        # Both variables start at 1.0
        return jnp.ones(self.n)

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Compute the objective function.

        The objective is not explicitly given in the SIF file, but from the
        element and group structure, it appears to be a least squares problem
        minimizing the constraint violations.
        """
        x1, x2 = y
        # From the SIF structure, minimize sum of squares of constraint violations
        g1 = x1**2 + x2 - 11.0
        g2 = x1 + x2**2 - 7.0
        return g1**2 + g2**2

    def equality_constraints(self, y, args):
        """Compute the equality constraints."""
        x1, x2 = y
        g1 = x1**2 + x2 - 11.0
        g2 = x1 + x2**2 - 7.0
        return jnp.array([g1, g2])

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # At the optimal solution, constraints should be satisfied, so objective = 0
        return jnp.array(0.0)
