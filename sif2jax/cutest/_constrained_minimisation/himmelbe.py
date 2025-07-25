import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HIMMELBE(AbstractConstrainedMinimisation):
    """A 3 variables problem by Himmelblau.

    Source: problem 30 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.

    See Buckley#88 (p. 65)

    SIF input: Ph. Toint, Dec 1989.

    classification: NQR2-AY-3-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 3  # 3 variables
    m_eq: int = 3  # 3 equality constraints
    m_ineq: int = 0  # no inequality constraints

    @property
    def y0(self):
        # Starting point from SIF file
        return jnp.array([-1.2, 2.0, 0.0])

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Compute the objective function.

        This is a feasibility problem with no explicit objective,
        so the objective is constant (zero).
        """
        return jnp.array(0.0)

    @property
    def bounds(self):
        """Returns the bounds on the variable y."""
        # No bounds specified in the SIF file (FR = free)
        return None

    def constraint(self, y):
        """Returns the constraints on the variable y."""
        x1, x2, x3 = y

        # Equality constraints: g(x) = 0
        # G1: 0.25 * (x1 + x2)^2 - x3 = 0
        g1 = 0.25 * (x1 + x2) ** 2 - x3

        # G2: -x1 + 1.0 = 0 => x1 = 1.0
        g2 = -x1 + 1.0

        # G3: -x2 + 1.0 = 0 => x2 = 1.0
        g3 = -x2 + 1.0

        eq_constraints = jnp.array([g1, g2, g3])

        # No inequality constraints
        ineq_constraints = None

        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        # From the constraints: x1 = 1.0, x2 = 1.0, x3 = 0.25*(1+1)^2 = 1.0
        return jnp.array([1.0, 1.0, 1.0])

    @property
    def expected_objective_value(self):
        # For a feasibility problem, the objective remains 0
        return jnp.array(0.0)
