import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS84(AbstractConstrainedMinimisation):
    """HS84 problem from Hock & Schittkowski collection.

    TODO: Human review needed - objective value has ~2% discrepancy with pycutest.

    Challenges encountered:
    - Understanding MA(i) notation in SIF files (means minus A(i))
    - Objective uses MA(2) through MA(6) which negates those coefficients
    - The problem has range constraints (0 <= g(x) <= r) that pycutest handles specially
    - Even after fixing signs, ~2% discrepancy remains - could be numerical precision
    - Product elements E(i) = X(1) * X(i+1) are used in objective and constraints

    Source: problem 84 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: A.R. Conn, March 1991.

    Classification: QQR2-AN-5-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 5

    @property
    def m(self):
        """Number of constraints."""
        return 3  # Three range constraints

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        x = y

        # Parameters
        a = jnp.array(
            [
                -24345.0,
                -8720288.849,
                150512.5253,
                -156.6950325,
                476470.3222,
                729482.8271,
                -145421.402,
                2931.1506,
                -40.427932,
                5106.192,
                15711.36,
                -155011.1084,
                4360.53352,
                12.9492344,
                10236.884,
                13176.786,
                -326669.5104,
                7390.68412,
                -27.8986976,
                16643.076,
                30988.146,
            ]
        )

        # Objective: a1 - a2*x1 - a3*x1*x2 - a4*x1*x3 - a5*x1*x4 - a6*x1*x5
        # Note: MA(i) in SIF means minus A(i)
        obj = (
            a[0]
            - a[1] * x[0]  # MA(2) means minus
            - a[2] * x[0] * x[1]  # MA(3) means minus
            - a[3] * x[0] * x[2]  # MA(4) means minus
            - a[4] * x[0] * x[3]  # MA(5) means minus
            - a[5] * x[0] * x[4]  # MA(6) means minus
        )

        return obj

    def constraint(self, y):
        """Compute the constraints."""
        x = y

        # Parameters
        a = jnp.array(
            [
                -24345.0,
                -8720288.849,
                150512.5253,
                -156.6950325,
                476470.3222,
                729482.8271,
                -145421.402,
                2931.1506,
                -40.427932,
                5106.192,
                15711.36,
                -155011.1084,
                4360.53352,
                12.9492344,
                10236.884,
                13176.786,
                -326669.5104,
                7390.68412,
                -27.8986976,
                16643.076,
                30988.146,
            ]
        )

        # Constraints are range constraints: 0 <= g(x) <= r
        # CON1: a7*x1 + a8*x1*x2 + a9*x1*x3 + a10*x1*x4 + a11*x1*x5
        g1 = (
            a[6] * x[0]
            + a[7] * x[0] * x[1]
            + a[8] * x[0] * x[2]
            + a[9] * x[0] * x[3]
            + a[10] * x[0] * x[4]
        )

        # CON2: a12*x1 + a13*x1*x2 + a14*x1*x3 + a15*x1*x4 + a16*x1*x5
        g2 = (
            a[11] * x[0]
            + a[12] * x[0] * x[1]
            + a[13] * x[0] * x[2]
            + a[14] * x[0] * x[3]
            + a[15] * x[0] * x[4]
        )

        # CON3: a17*x1 + a18*x1*x2 + a19*x1*x3 + a20*x1*x4 + a21*x1*x5
        g3 = (
            a[16] * x[0]
            + a[17] * x[0] * x[1]
            + a[18] * x[0] * x[2]
            + a[19] * x[0] * x[3]
            + a[20] * x[0] * x[4]
        )

        # Range constraints: 0 <= g(x) <= r
        # pycutest seems to return g(x) directly for range constraints
        constraints = jnp.array([g1, g2, g3])

        # Return as inequality constraints
        return None, constraints

    def y0(self):
        """Initial guess."""
        return jnp.array([2.52, 2.0, 37.5, 9.25, 6.8])

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds."""
        lower = jnp.array([0.0, 1.2, 20.0, 9.0, 6.5])
        upper = jnp.array([1000.0, 2.4, 60.0, 9.3, 7.0])
        return lower, upper

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None
