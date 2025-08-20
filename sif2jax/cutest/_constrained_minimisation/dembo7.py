import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class DEMBO7(AbstractConstrainedMinimisation):
    """A 7 stage membrane separation model.

    Source: problem 7 in
    R.S. Dembo,
    "A set of geometric programming test problems and their solutions",
    Mathematical Programming, 17, 192-213, 1976.

    SIF input: A. R. Conn, June 1993.

    classification: QOR2-MN-16-20
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 16  # 16 variables
    m_eq: int = 0  # no equality constraints
    m_ineq: int = 20  # 20 inequality constraints (including range constraint)

    @property
    def y0(self):
        # Starting point from SIF file
        return jnp.array(
            [
                0.8,
                0.83,
                0.85,
                0.87,
                0.90,
                0.10,
                0.12,
                0.19,
                0.25,
                0.29,
                512.0,
                13.1,
                71.8,
                640.0,
                650.0,
                5.7,
            ]
        )

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Nonlinear objective function."""
        # Linear terms: 1.262626*(X12 + X13 + X14 + X15 + X16)
        # From GROUP USES: E1-E5 with coefficient -1.231060
        # E1: 2PR(X1, X12), E2: 2PR(X2, X13), etc.
        obj = 1.262626 * (y[11] + y[12] + y[13] + y[14] + y[15])
        # Add nonlinear terms
        obj += -1.231060 * (
            y[0] * y[11]  # E1
            + y[1] * y[12]  # E2
            + y[2] * y[13]  # E3
            + y[3] * y[14]  # E4
            + y[4] * y[15]  # E5
        )
        return obj

    @property
    def bounds(self):
        """Bounds on variables."""
        lower = jnp.array(
            [
                0.1,  # X1
                0.1,  # X2
                0.1,  # X3
                0.1,  # X4
                0.9,  # X5
                0.0001,  # X6
                0.1,  # X7
                0.1,  # X8
                0.1,  # X9
                0.1,  # X10
                1.0,  # X11
                0.000001,  # X12
                1.0,  # X13
                500.0,  # X14
                500.0,  # X15
                0.000001,  # X16
            ]
        )
        upper = jnp.array(
            [
                0.9,  # X1
                0.9,  # X2
                0.9,  # X3
                0.9,  # X4
                1.0,  # X5
                0.1,  # X6
                0.9,  # X7
                0.9,  # X8
                0.9,  # X9
                0.9,  # X10
                1000.0,  # X11
                500.0,  # X12
                500.0,  # X13
                1000.0,  # X14
                1000.0,  # X15
                500.0,  # X16
            ]
        )
        return lower, upper

    def constraint(self, y):
        """Returns the constraints on the variable y.

        20 inequality constraints.
        """
        # No equality constraints
        eq_constraints = None

        # Inequality constraints
        # Pycutest convention: constraints are in form c <= 0
        # We need to return -c to get c >= 0 form for sif2jax
        c = jnp.zeros(20)

        # C0: Range constraint 50 <= obj <= 250
        # Pycutest transforms to 0 <= (obj - 50) <= 200
        # Pycutest reports (obj - 50) as the constraint value
        obj_val = 1.262626 * (y[11] + y[12] + y[13] + y[14] + y[15]) - 1.231060 * (
            y[0] * y[11] + y[1] * y[12] + y[2] * y[13] + y[3] * y[14] + y[4] * y[15]
        )
        # Match pycutest: report (obj - 50) which should be in [0, 200]
        # Since we need c >= 0 form and pycutest uses c <= 0, we negate later
        c = c.at[0].set(obj_val - 50.0)

        # C1: 0.975*X1 + E6*0.034750 + E7*(-0.00975) <= 1.0
        # E6: QT(X1, X6) = X1/X6,  E7: SQQT(X1, X6) = X1^2/X6
        e6 = y[0] / jnp.maximum(y[5], 1e-10)
        e7 = y[0] ** 2 / jnp.maximum(y[5], 1e-10)
        c = c.at[1].set(1.0 - 0.975 * y[0] - 0.034750 * e6 + 0.00975 * e7)

        # C2: 0.975*X2 + E8*0.034750 + E9*(-0.00975) <= 1.0
        # E8: QT(X2, X7), E9: SQQT(X2, X7)
        e8 = y[1] / jnp.maximum(y[6], 1e-10)
        e9 = y[1] ** 2 / jnp.maximum(y[6], 1e-10)
        c = c.at[2].set(1.0 - 0.975 * y[1] - 0.034750 * e8 + 0.00975 * e9)

        # C3: 0.975*X3 + E10*0.034750 + E11*(-0.00975) <= 1.0
        # E10: QT(X3, X8), E11: SQQT(X3, X8)
        e10 = y[2] / jnp.maximum(y[7], 1e-10)
        e11 = y[2] ** 2 / jnp.maximum(y[7], 1e-10)
        c = c.at[3].set(1.0 - 0.975 * y[2] - 0.034750 * e10 + 0.00975 * e11)

        # C4: 0.975*X4 + E12*0.034750 + E13*(-0.00975) <= 1.0
        # E12: QT(X4, X9), E13: SQQT(X4, X9)
        e12 = y[3] / jnp.maximum(y[8], 1e-10)
        e13 = y[3] ** 2 / jnp.maximum(y[8], 1e-10)
        c = c.at[4].set(1.0 - 0.975 * y[3] - 0.034750 * e12 + 0.00975 * e13)

        # C5: 0.975*X5 + E14*0.034750 + E15*(-0.00975) <= 1.0
        # E14: QT(X5, X10), E15: SQQT(X5, X10)
        e14 = y[4] / jnp.maximum(y[9], 1e-10)
        e15 = y[4] ** 2 / jnp.maximum(y[9], 1e-10)
        c = c.at[5].set(1.0 - 0.975 * y[4] - 0.034750 * e14 + 0.00975 * e15)

        # C6: E16 + E17 - E18 <= 1.0
        # E16: QT(X6, X7), E17: QTQT(X1, X7, X12, X11), E18: QTQT(X6, X7, X12, X11)
        e16 = y[5] / jnp.maximum(y[6], 1e-10)
        e17 = (y[0] * y[11]) / jnp.maximum(
            y[6] * y[10], 1e-10
        )  # QTQT(X1, X7, X12, X11)
        e18 = (y[5] * y[11]) / jnp.maximum(
            y[6] * y[10], 1e-10
        )  # QTQT(X6, X7, X12, X11)
        c = c.at[6].set(1.0 - e16 - e17 + e18)

        # C7: -0.002*X13 + E19 + E20*0.002 + E21*0.002 - E22*0.002 <= 1.0
        # E19: QT(X7, X8), E20-E22: 2PRRC elements
        e19 = y[6] / jnp.maximum(y[7], 1e-10)  # QT(X7, X8)
        e20 = (y[6] * y[11]) / jnp.maximum(y[7], 1e-10)  # 2PRRC(X7, X12, X8)
        e21 = (y[1] * y[12]) / jnp.maximum(y[7], 1e-10)  # 2PRRC(X2, X13, X8)
        e22 = (y[0] * y[11]) / jnp.maximum(y[7], 1e-10)  # 2PRRC(X1, X12, X8)
        c = c.at[7].set(
            1.0 + 0.002 * y[12] - e19 - 0.002 * e20 - 0.002 * e21 + 0.002 * e22
        )

        # C8: X8 + X9 + E23*0.002 + E24*0.002 - E25*0.002 - E26*0.002 <= 1.0
        # E23: 2PR(X8,X13), E24: 2PR(X3,X14), E25: 2PR(X2,X13), E26: 2PR(X9,X14)
        e23 = y[7] * y[12]
        e24 = y[2] * y[13]
        e25 = y[1] * y[12]
        e26 = y[8] * y[13]
        c = c.at[8].set(
            1.0 - y[7] - y[8] - 0.002 * e23 - 0.002 * e24 + 0.002 * e25 + 0.002 * e26
        )

        # C9: E27 + E28 + E29*500 - E30*500 - E31 <= 1.0
        # E27: QT(X9,X3), E28: QTQT(X4,X3,X15,X14), E29: QTRC(X10,X3,X14)
        # E30: QTRC(X9,X3,X14), E31: QTQT(X8,X3,X15,X14)
        e27 = y[8] / jnp.maximum(y[2], 1e-10)
        e28 = (y[3] * y[14]) / jnp.maximum(y[2] * y[13], 1e-10)
        e29 = y[9] / jnp.maximum(y[2] * y[13], 1e-10)
        e30 = y[8] / jnp.maximum(y[2] * y[13], 1e-10)
        e31 = (y[7] * y[14]) / jnp.maximum(y[2] * y[13], 1e-10)
        c = c.at[9].set(1.0 - e27 - e28 - 500.0 * e29 + 500.0 * e30 + e31)

        # C10: E32 + E33 + E34*500 - E35 - E36*500 <= 1.0
        # E32: QTQT(X5,X4,X16,X15), E33: QT(X10,X4), E34: INV(X15)
        # E35: QT(X16,X15), E36: QTRC(X10,X4,X15)
        e32 = (y[4] * y[15]) / jnp.maximum(y[3] * y[14], 1e-10)
        e33 = y[9] / jnp.maximum(y[3], 1e-10)
        e34 = 1.0 / jnp.maximum(y[14], 1e-10)
        e35 = y[15] / jnp.maximum(y[14], 1e-10)
        e36 = y[9] / jnp.maximum(y[3] * y[14], 1e-10)
        c = c.at[10].set(1.0 - e32 - e33 - 500.0 * e34 + e35 + 500.0 * e36)

        # C11: 0.002*X16 + E37*0.9 - E38*0.002 <= 1.0
        # E37: INV(X4), E38: 2PRRC(X5,X16,X4) = X5*X16/X4
        e37 = 1.0 / jnp.maximum(y[3], 1e-10)
        e38 = (y[4] * y[15]) / jnp.maximum(y[3], 1e-10)
        c = c.at[11].set(1.0 - 0.002 * y[15] - 0.9 * e37 + 0.002 * e38)

        # C12: 0.002*X11 - 0.002*X12 <= 1.0 (no elements in GROUP USES)
        c = c.at[12].set(1.0 - 0.002 * y[10] + 0.002 * y[11])

        # C13: E39 <= 1.0, E39: QT(X12,X11)
        e39 = y[11] / jnp.maximum(y[10], 1e-10)
        c = c.at[13].set(1.0 - e39)

        # C14: E40 <= 1.0, E40: QT(X4,X5)
        e40 = y[3] / jnp.maximum(y[4], 1e-10)
        c = c.at[14].set(1.0 - e40)

        # C15: E41 <= 1.0, E41: QT(X3,X4)
        e41 = y[2] / jnp.maximum(y[3], 1e-10)
        c = c.at[15].set(1.0 - e41)

        # C16: E42 <= 1.0, E42: QT(X2,X3)
        e42 = y[1] / jnp.maximum(y[2], 1e-10)
        c = c.at[16].set(1.0 - e42)

        # C17: E43 <= 1.0, E43: QT(X1,X2)
        e43 = y[0] / jnp.maximum(y[1], 1e-10)
        c = c.at[17].set(1.0 - e43)

        # C18: E44 <= 1.0, E44: QT(X9,X10)
        e44 = y[8] / jnp.maximum(y[9], 1e-10)
        c = c.at[18].set(1.0 - e44)

        # C19: E45 <= 1.0, E45: QT(X8,X9)
        e45 = y[7] / jnp.maximum(y[8], 1e-10)
        c = c.at[19].set(1.0 - e45)

        # Sign convention:
        # - C0 is a range constraint, pycutest reports (obj - 50) directly
        # - C1-C19 are <= constraints, pycutest uses c <= 0 form
        # We need to negate C1-C19 but not C0
        ineq_constraints = c.at[1:].set(-c[1:])

        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file: *LO SOLTN 174.788807
        return jnp.array(174.788807)
