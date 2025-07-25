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
        x1, x2 = y
        # Equality constraints: g(x) = 0
        g1 = x1**2 + x2 - 11.0
        g2 = x1 + x2**2 - 7.0
        eq_constraints = jnp.array([g1, g2])
        # No inequality constraints
        ineq_constraints = None
        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # For a feasibility problem, the objective remains 0
        return jnp.array(0.0)
