import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS87(AbstractConstrainedMinimisation):
    """Hock and Schittkowski problem 87.

    Optimization of an electrical network (EDF) by P. Huard.

    Source: problem 87 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: Nick Gould, August 1991.

    classification OOI2-MN-6-4
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(6)

    @property
    def y0(self):
        """Initial guess."""
        # Using HS87SOL values as starting point (pycutest does the same)
        return jnp.array(
            [107.8119, 196.3186, 373.8307, 420.0, 21.30713, 0.153292], dtype=jnp.float64
        )

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Objective function."""
        del args
        x1, x2 = y[0], y[1]

        # F1(X1): piecewise linear function
        # If X1 < 300: F1 = 30*X1
        # If X1 >= 300: F1 = 31*X1
        f1 = jnp.where(x1 < 300.0, 30.0 * x1, 31.0 * x1)

        # F2(X2): piecewise linear function
        # If X2 < 100: F2 = 28*X2
        # If 100 <= X2 < 200: F2 = 29*X2
        # If X2 >= 200: F2 = 30*X2
        f2 = jnp.where(
            x2 < 100.0, 28.0 * x2, jnp.where(x2 < 200.0, 29.0 * x2, 30.0 * x2)
        )

        return f1 + f2

    @property
    def bounds(self):
        """Variable bounds."""
        # From BOUNDS section
        lower = jnp.array([0.0, 0.0, 340.0, 340.0, -1000.0, 0.0], dtype=jnp.float64)
        upper = jnp.array(
            [400.0, 1000.0, 420.0, 420.0, 10000.0, 0.5236], dtype=jnp.float64
        )
        return lower, upper

    def constraint(self, y):
        """Constraint functions."""
        x1, x2, x3, x4, x5, x6 = y[0], y[1], y[2], y[3], y[4], y[5]

        # Constants from SIF file
        A = 131.078
        B = 1.48577
        C = 0.90798
        D = jnp.cos(1.47588)
        E = jnp.sin(1.47588)
        F = 1.48577

        # Derived constants
        one_over_A = 1.0 / A
        minus_one_over_A = -one_over_A
        CD_over_A = (C / A) * D
        CE_over_A = (C / A) * E

        # Element values:
        # C1E1 (COS): X3 * X4 * cos(X6 + P) where P = -F
        c1e1 = x3 * x4 * jnp.cos(x6 - F)
        # C1E2 (SQUARE): X3^2
        c1e2 = x3 * x3

        # C2E1 (COS): X3 * X4 * cos(X6 + P) where P = B
        c2e1 = x3 * x4 * jnp.cos(x6 + B)
        # C2E2 (SQUARE): X4^2
        c2e2 = x4 * x4

        # C3E1 (SIN): X3 * X4 * sin(X6 + P) where P = B
        c3e1 = x3 * x4 * jnp.sin(x6 + B)
        # C3E2 (SQUARE): X4^2
        c3e2 = x4 * x4

        # C4E1 (SIN): X3 * X4 * sin(X6 + P) where P = -B
        c4e1 = x3 * x4 * jnp.sin(x6 - B)
        # C4E2 (SQUARE): X3^2
        c4e2 = x3 * x3

        # Constraints from GROUPS + CONSTANTS + GROUP USES:
        # The CONSTANTS section adds -300 to C1 and -200 to C4
        # But these need to be negated when forming the constraint equation
        # C1: -X1 + 300 + C1E1*(-1/A) + C1E2*(CD/A) = 0
        c1 = -x1 + 300.0 + c1e1 * minus_one_over_A + c1e2 * CD_over_A

        # C2: -X2 + 0 + C2E1*(-1/A) + C2E2*(CD/A) = 0
        c2 = -x2 + c2e1 * minus_one_over_A + c2e2 * CD_over_A

        # C3: -X5 + 0 + C3E1*(-1/A) + C3E2*(CE/A) = 0
        c3 = -x5 + c3e1 * minus_one_over_A + c3e2 * CE_over_A

        # C4: 0 + 200 + C4E1*(1/A) + C4E2*(CE/A) = 0
        c4 = 200.0 + c4e1 * one_over_A + c4e2 * CE_over_A

        # All constraints are equality constraints (type E in SIF)
        equalities = jnp.array([c1, c2, c3, c4])

        return equalities, None

    @property
    def expected_result(self):
        """Expected result from SIF file."""
        # From HS87SOL section
        return jnp.array(
            [107.8119, 196.3186, 373.8307, 420.0, 21.30713, 0.153292], dtype=jnp.float64
        )

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # From OBJECT BOUND section
        return jnp.array(8927.5977)
