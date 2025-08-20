import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class DIPIGRI(AbstractConstrainedMinimisation):
    """A problem proposed by Di Pillo and Grippo.

    Source:
    G. Di Pillo and L. Grippo,
    "An new augmented Lagrangian function for inequality constraints
    in nonlinear programming problems",
    JOTA, vol. 36, pp. 495-519, 1982.

    SIF input: Ph. Toint, June 1990.

    classification: OOR2-AN-7-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 7  # 7 variables
    m_eq: int = 0  # no equality constraints
    m_ineq: int = 4  # 4 inequality constraints

    @property
    def y0(self):
        # Starting point from SIF file
        return jnp.array([1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0])

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Nonlinear objective function."""
        x1, x2, x3, x4, x5, x6, x7 = y

        # Element evaluations
        o1 = (x1 - 10.0) ** 2
        o2 = 5.0 * (x2 - 12.0) ** 2
        o3 = x3**4
        o4 = 3.0 * (x4 - 11.0) ** 2
        o5 = 10.0 * x5**6
        o6 = 7.0 * x6**2 + x7**4 - 4.0 * x6 * x7

        # Linear terms
        linear = -10.0 * x6 - 8.0 * x7

        return o1 + o2 + o3 + o4 + o5 + o6 + linear

    @property
    def bounds(self):
        """No bounds on variables."""
        return None

    def constraint(self, y):
        """Returns the constraints on the variable y.

        4 inequality constraints (g(x) >= 0).
        """
        x1, x2, x3, x4, x5, x6, x7 = y

        # No equality constraints
        eq_constraints = None

        # Inequality constraints (must be >= 0)
        # From SIF: constraints are in form Ax + f(x) <= b which becomes
        # b - Ax - f(x) >= 0. But pycutest uses opposite sign convention, so negate

        # C1: 2*x1^2 + 3*x2^4 + x3 + 4*x4^2 + 5*x5 - 127 >= 0 (negated)
        c1 = 2.0 * x1**2 + 3.0 * x2**4 + x3 + 4.0 * x4**2 + 5.0 * x5 - 127.0

        # C2: 7*x1 + 3*x2 + 10*x3^2 + x4 - x5 - 282 >= 0 (negated)
        c2 = 7.0 * x1 + 3.0 * x2 + 10.0 * x3**2 + x4 - x5 - 282.0

        # C3: 23*x1 + x2^2 + 6*x6^2 - 8*x7 - 196 >= 0 (negated)
        c3 = 23.0 * x1 + x2**2 + 6.0 * x6**2 - 8.0 * x7 - 196.0

        # C4: 4*x1^2 + x2^2 - 3*x1*x2 + 2*x3^2 + 5*x6 - 11*x7 >= 0 (negated)
        c4 = 4.0 * x1**2 + x2**2 - 3.0 * x1 * x2 + 2.0 * x3**2 + 5.0 * x6 - 11.0 * x7

        ineq_constraints = jnp.array([c1, c2, c3, c4])

        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment: *LO SOLTN 680.630
        return jnp.array(680.630)
