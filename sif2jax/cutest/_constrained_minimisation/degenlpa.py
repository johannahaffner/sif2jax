import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# Constraint coefficient matrix for DEGENLPA
# Rows: constraints C1-C15
# Columns: variables X1-X20
# fmt: off
_A = jnp.array([
    # C1
    [1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # C2
    [-1.0, 300.0, 0.09, 0.03, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # C3
    [0.326, -101.0, 0, 0, 200.0, 0.06, 0.02, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # C4
    [0.0066667, 0, -1.03, 0, 0, 200.0, 0, 0.06, 0.02, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # C5
    [6.6667e-4, 0, 0, -1.01, 0, 0, 200.0, 0, 0.06, 0.02,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # C6
    [0, 0.978, 0, 0, -201.0, 0, 0, 0, 0, 0,
     100.0, 0.03, 0.01, 0, 0, 0, 0, 0, 0, 0],
    # C7
    [0, 0.01, 0.489, 0, 0, -101.03, 0, 0, 0, 0,
     0, 100.0, 0, 0.03, 0.01, 0, 0, 0, 0, 0],
    # C8
    [0, 0.001, 0, 0.489, 0, 0, -101.03, 0, 0, 0,
     0, 0, 100.0, 0, 0.03, 0.01, 0, 0, 0, 0],
    # C9
    [0, 0, 0.001, 0.01, 0, 0, 0, 0, -1.04, 0,
     0, 0, 0, 0, 100.0, 0, 0, 0.03, 0.01, 0],
    # C10
    [0, 0, 0.02, 0, 0, 0, 0, -1.06, 0, 0,
     0, 0, 0, 100.0, 0, 0, 0.03, 0, 0.01, 0],
    # C11
    [0, 0, 0, 0.002, 0, 0, 0, 0, 0, -1.02,
     0, 0, 0, 0, 0, 100.0, 0, 0, 0.03, 0.01],
    # C12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -2.5742e-6, 0, 0.00252, 0, 0, -0.61975, 0, 0, 0, 1.03],
    # C13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -0.00257, 0.25221, 0, -6.2, 0, 0, 1.09, 0, 0, 0],
    # C14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0.00629, -0.20555, -4.1106, 0, 101.04, 505.1, 0, 0, -256.72, 0],
    # C15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0.00841, -0.08406, -0.20667, 0, 20.658, 0, 1.07, -10.5, 0],
])
# fmt: on

# Right-hand side vector for constraints
_B = jnp.array([0.70785, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


class DEGENLPA(AbstractConstrainedMinimisation):
    """A small linear program with some degeneracy.

    Source:
    T.C.T. Kotiah and D.I. Steinberg,
    "Occurences of cycling and other phenomena arising in a class of
    linear programming models",
    Communications of the ACM, vol. 20, pp. 107-112, 1977.

    SIF input: Ph. Toint, Aug 1990.

    classification: LLR2-AN-20-15
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 20  # 20 variables
    m_eq: int = 15  # 15 equality constraints
    m_ineq: int = 0  # no inequality constraints

    @property
    def y0(self):
        # All variables start at 1.0 (START POINT: XV DEGENLPA 'DEFAULT' 1.0)
        return jnp.ones(20)

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Linear objective function."""
        # From GROUPS section N OBJ entries
        obj = (
            0.01 * y[1]  # X2
            + 33.333 * y[2]  # X3
            + 100.0 * y[3]  # X4
            + 0.01 * y[4]  # X5
            + 33.343 * y[5]  # X6
            + 100.01 * y[6]  # X7
            + 33.333 * y[7]  # X8
            + 133.33 * y[8]  # X9
            + 100.0 * y[9]  # X10
        )
        return obj

    @property
    def bounds(self):
        """Bounds: 0 <= x <= 1."""
        # From BOUNDS: XU DEGENLPA 'DEFAULT' 1.0
        # Default lower bound is 0 for LP problems
        lower = jnp.zeros(self.n)
        upper = jnp.ones(self.n)
        return lower, upper

    def constraint(self, y):
        """Returns the constraints on the variable y.

        15 equality constraints from C1 to C15.
        """
        # Compute constraints using module-level matrix: A @ y - b
        eq_constraints = _A @ y - _B

        # No inequality constraints
        ineq_constraints = None

        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment: *LO SOLTN               -3.06435
        return jnp.array(-3.06435)
