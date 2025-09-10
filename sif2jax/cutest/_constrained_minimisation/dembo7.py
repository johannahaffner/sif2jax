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

        # Vectorized constraint computation
        # C0: Range constraint 50 <= obj <= 250
        obj_val = 1.262626 * (y[11] + y[12] + y[13] + y[14] + y[15]) - 1.231060 * (
            y[0] * y[11] + y[1] * y[12] + y[2] * y[13] + y[3] * y[14] + y[4] * y[15]
        )
        c0 = obj_val - 50.0

        # C1-C5: Vectorized computation for similar constraints
        # Pattern: 0.975*Xi + QT(Xi, Xi+5)*0.034750 + SQQT(Xi, Xi+5)*(-0.00975) <= 1.0
        x_vals = y[:5]  # X1 through X5
        denom_vals = jnp.maximum(y[5:10], 1e-10)  # X6 through X10
        qt_vals = x_vals / denom_vals  # QT terms
        sqqt_vals = x_vals**2 / denom_vals  # SQQT terms
        c1_5 = 1.0 - 0.975 * x_vals - 0.034750 * qt_vals + 0.00975 * sqqt_vals

        # C6: E16 + E17 - E18 <= 1.0
        e16 = y[5] / jnp.maximum(y[6], 1e-10)
        e17 = (y[0] * y[11]) / jnp.maximum(y[6] * y[10], 1e-10)
        e18 = (y[5] * y[11]) / jnp.maximum(y[6] * y[10], 1e-10)
        c6 = 1.0 - e16 - e17 + e18

        # C7: -0.002*X13 + E19 + E20*0.002 + E21*0.002 - E22*0.002 <= 1.0
        e19 = y[6] / jnp.maximum(y[7], 1e-10)
        e20 = (y[6] * y[11]) / jnp.maximum(y[7], 1e-10)
        e21 = (y[1] * y[12]) / jnp.maximum(y[7], 1e-10)
        e22 = (y[0] * y[11]) / jnp.maximum(y[7], 1e-10)
        c7 = 1.0 + 0.002 * y[12] - e19 - 0.002 * e20 - 0.002 * e21 + 0.002 * e22

        # C8: X8 + X9 + E23*0.002 + E24*0.002 - E25*0.002 - E26*0.002 <= 1.0
        e23 = y[7] * y[12]
        e24 = y[2] * y[13]
        e25 = y[1] * y[12]
        e26 = y[8] * y[13]
        c8 = 1.0 - y[7] - y[8] - 0.002 * e23 - 0.002 * e24 + 0.002 * e25 + 0.002 * e26

        # C9: E27 + E28 + E29*500 - E30*500 - E31 <= 1.0
        e27 = y[8] / jnp.maximum(y[2], 1e-10)
        e28 = (y[3] * y[14]) / jnp.maximum(y[2] * y[13], 1e-10)
        e29 = y[9] / jnp.maximum(y[2] * y[13], 1e-10)
        e30 = y[8] / jnp.maximum(y[2] * y[13], 1e-10)
        e31 = (y[7] * y[14]) / jnp.maximum(y[2] * y[13], 1e-10)
        c9 = 1.0 - e27 - e28 - 500.0 * e29 + 500.0 * e30 + e31

        # C10: E32 + E33 + E34*500 - E35 - E36*500 <= 1.0
        e32 = (y[4] * y[15]) / jnp.maximum(y[3] * y[14], 1e-10)
        e33 = y[9] / jnp.maximum(y[3], 1e-10)
        e34 = 1.0 / jnp.maximum(y[14], 1e-10)
        e35 = y[15] / jnp.maximum(y[14], 1e-10)
        e36 = y[9] / jnp.maximum(y[3] * y[14], 1e-10)
        c10 = 1.0 - e32 - e33 - 500.0 * e34 + e35 + 500.0 * e36

        # C11: 0.002*X16 + E37*0.9 - E38*0.002 <= 1.0
        e37 = 1.0 / jnp.maximum(y[3], 1e-10)
        e38 = (y[4] * y[15]) / jnp.maximum(y[3], 1e-10)
        c11 = 1.0 - 0.002 * y[15] - 0.9 * e37 + 0.002 * e38

        # C12: 0.002*X11 - 0.002*X12 <= 1.0
        c12 = 1.0 - 0.002 * y[10] + 0.002 * y[11]

        # C13-C19: Vectorized QT constraints
        # C13: QT(X12,X11), C14: QT(X4,X5), C15: QT(X3,X4), C16: QT(X2,X3)
        # C17: QT(X1,X2), C18: QT(X9,X10), C19: QT(X8,X9)
        qt_num = jnp.array([y[11], y[3], y[2], y[1], y[0], y[8], y[7]])
        qt_denom = jnp.array([y[10], y[4], y[3], y[2], y[1], y[9], y[8]])
        c13_19 = 1.0 - qt_num / jnp.maximum(qt_denom, 1e-10)

        # Concatenate all constraints
        c_all = jnp.concatenate(
            [
                jnp.array([c0]),  # C0
                c1_5,  # C1-C5
                jnp.array([c6, c7, c8, c9, c10, c11, c12]),  # C6-C12
                c13_19,  # C13-C19
            ]
        )

        # Sign convention: negate C1-C19 but not C0
        ineq_constraints = jnp.concatenate(
            [
                jnp.array([c_all[0]]),  # C0 unchanged
                -c_all[1:],  # C1-C19 negated
            ]
        )

        return eq_constraints, ineq_constraints

    @property
    def expected_result(self):
        # The optimal solution is not explicitly given in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # From the SIF file: *LO SOLTN 174.788807
        return jnp.array(174.788807)
