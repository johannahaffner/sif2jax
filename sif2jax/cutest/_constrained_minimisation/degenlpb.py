import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class DEGENLPB(AbstractConstrainedMinimisation):
    """A linear program with some degeneracy.

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
        # All variables start at 1.0 (START POINT: XV DEGENLPB 'DEFAULT' 1.0)
        return jnp.ones(20)

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Linear objective function."""
        # From GROUPS section N OBJ entries (note: negative coefficients)
        obj = (
            -0.01 * y[1]  # X2
            + (-33.333) * y[2]  # X3
            + (-100.0) * y[3]  # X4
            + (-0.01) * y[4]  # X5
            + (-33.343) * y[5]  # X6
            + (-100.01) * y[6]  # X7
            + (-33.333) * y[7]  # X8
            + (-133.33) * y[8]  # X9
            + (-100.0) * y[9]  # X10
        )
        return obj

    @property
    def bounds(self):
        """Bounds: 0 <= x <= 1."""
        # From BOUNDS: XU DEGENLPB 'DEFAULT' 1.0
        # Default lower bound is 0 for LP problems
        lower = jnp.zeros(self.n)
        upper = jnp.ones(self.n)
        return lower, upper

    def constraint(self, y):
        """Returns the constraints on the variable y.

        15 equality constraints from C1 to C15.
        """
        # Build the constraint matrix A and vector b
        # Constraint equations: A @ y = b

        # Initialize constraints array
        eq_constraints = jnp.zeros(15)

        # C1: coefficients from the SIF file
        eq_constraints = eq_constraints.at[0].set(
            1.0 * y[0]
            + 2.0 * y[1]
            + 2.0 * y[2]
            + 2.0 * y[3]
            + 1.0 * y[4]
            + 2.0 * y[5]
            + 2.0 * y[6]
            + 1.0 * y[7]
            + 2.0 * y[8]
            + 1.0 * y[9]
            - 0.70785
        )

        # C2
        eq_constraints = eq_constraints.at[1].set(
            -1.0 * y[0] + 300.0 * y[1] + 0.09 * y[2] + 0.03 * y[3]
        )

        # C3
        eq_constraints = eq_constraints.at[2].set(
            0.326 * y[0] + (-101.0) * y[1] + 200.0 * y[4] + 0.06 * y[5] + 0.02 * y[6]
        )

        # C4
        eq_constraints = eq_constraints.at[3].set(
            0.0066667 * y[0] + (-1.03) * y[2] + 200.0 * y[5] + 0.06 * y[7] + 0.02 * y[8]
        )

        # C5
        eq_constraints = eq_constraints.at[4].set(
            6.6667e-4 * y[0] + (-1.01) * y[3] + 200.0 * y[6] + 0.06 * y[8] + 0.02 * y[9]
        )

        # C6
        eq_constraints = eq_constraints.at[5].set(
            0.978 * y[1] + (-201.0) * y[4] + 100.0 * y[10] + 0.03 * y[11] + 0.01 * y[12]
        )

        # C7
        eq_constraints = eq_constraints.at[6].set(
            0.01 * y[1]
            + 0.489 * y[2]
            + (-101.03) * y[5]
            + 100.0 * y[11]
            + 0.03 * y[13]
            + 0.01 * y[14]
        )

        # C8
        eq_constraints = eq_constraints.at[7].set(
            0.001 * y[1]
            + 0.489 * y[3]
            + (-101.03) * y[6]
            + 100.0 * y[12]
            + 0.03 * y[14]
            + 0.01 * y[15]
        )

        # C9
        eq_constraints = eq_constraints.at[8].set(
            0.001 * y[2]
            + 0.01 * y[3]
            + (-1.04) * y[8]
            + 100.0 * y[14]
            + 0.03 * y[17]
            + 0.01 * y[18]
        )

        # C10
        eq_constraints = eq_constraints.at[9].set(
            0.02 * y[2] + (-1.06) * y[7] + 100.0 * y[13] + 0.03 * y[16] + 0.01 * y[18]
        )

        # C11
        eq_constraints = eq_constraints.at[10].set(
            0.002 * y[3] + (-1.02) * y[9] + 100.0 * y[15] + 0.03 * y[18] + 0.01 * y[19]
        )

        # C12
        eq_constraints = eq_constraints.at[11].set(
            (-2.5742e-6) * y[10] + 0.00252 * y[12] + (-0.61975) * y[15] + 1.03 * y[19]
        )

        # C13
        eq_constraints = eq_constraints.at[12].set(
            (-0.00257) * y[10] + 0.25221 * y[11] + (-6.2) * y[13] + 1.09 * y[16]
        )

        # C14
        eq_constraints = eq_constraints.at[13].set(
            0.00629 * y[10]
            + (-0.20555) * y[11]
            + (-4.1106) * y[12]
            + 101.04 * y[14]
            + 505.1 * y[15]
            + (-256.72) * y[18]
        )

        # C15
        eq_constraints = eq_constraints.at[14].set(
            0.00841 * y[11]
            + (-0.08406) * y[12]
            + (-0.20667) * y[13]
            + 20.658 * y[15]
            + 1.07 * y[17]
            + (-10.5) * y[18]
        )

        # No inequality constraints
        ineq_constraints = None

        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file comment: *LO SOLTN 3.06435
        return jnp.array(3.06435)
