import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class DEMYMALO(AbstractConstrainedMinimisation):
    """A nonlinear minmax problem in two variables.

    Source:
    V.F. Demynanov and V.N. Malozemov
    "Introduction to Minimax"
    Wiley, 1974

    SIF input: Ph. Toint, Nov 1993.

    classification: LQR2-AN-3-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 3  # 2 original variables + 1 auxiliary variable U
    m_eq: int = 0  # no equality constraints
    m_ineq: int = 3  # 3 inequality constraints

    @property
    def y0(self):
        # Starting point from SIF file
        return jnp.array([1.0, 1.0, 0.0])  # X1, X2, U (U starts at 0)

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Linear objective function: minimize U."""
        x1, x2, u = y
        return u

    @property
    def bounds(self):
        """No bounds on variables."""
        return None

    def constraint(self, y):
        """Returns the constraints on the variable y.

        3 inequality constraints (g(x) >= 0).
        Reformulated from minmax: U >= f_i(x) becomes U - f_i(x) >= 0
        """
        x1, x2, u = y

        # No equality constraints
        eq_constraints = None

        # Inequality constraints (must be >= 0)
        # Pycutest uses opposite sign convention
        # F1: (5*X1 + X2) - U >= 0 (negated)
        f1 = (5.0 * x1 + x2) - u

        # F2: (-5*X1 + X2) - U >= 0 (negated)
        f2 = (-5.0 * x1 + x2) - u

        # F3: (X1^2 + X2^2 + 4*X2) - U >= 0 (negated)
        f3 = (x1**2 + x2**2 + 4.0 * x2) - u

        ineq_constraints = jnp.array([f1, f2, f3])

        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment: *LO SOLTN -3.0
        return jnp.array(-3.0)
