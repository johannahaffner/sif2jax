import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made: Fixed E7 group formulation (added missing SO and SS terms),
# fixed constraint signs (E23, E24, E25)
# Suspected issues: Gradient test shows systematic factor of ~1.573 (close to π/2)
# difference for variables used in E16
# Additional resources needed: Clarification on GROUP TYPE implementation or
# element scaling factors
class AVION2(AbstractConstrainedMinimisation):
    """Dassault France avion (airplane design) problem.

    An aircraft design optimization problem with 49 variables representing
    various design parameters and performance characteristics.

    SIF input: A. R. Conn, June 1993.

    Classification: OLR2-RN-49-15
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 49

    @property
    def m(self):
        """Number of constraints."""
        return 15

    def objective(self, y, args):
        """Compute the objective function."""
        del args

        # Extract variables
        sr, lr, pk, ef, sx, lx, sd, sk, st, sf = y[0:10]
        lf, am, ca, cb, so, ss, impder, impk, impfus = y[10:19]
        qi, pt, mv, mc, md, pd, ns, vs, cr, pm = y[19:29]
        dv, mz, vn, qv, qf, imptrain, impmot, impnmot = y[29:37]
        imppet, imppil, impcan, impsna, ms, el, de, ds = y[37:45]
        impvoil, nm, np, ng = y[45:49]

        # Objective is sum of squares of constraint violations (SQUARE group type)
        obj = jnp.array(0.0)

        # E4: SK^2 - 0.01 * EL1 (2PR: PK * SR)
        e4 = sk - 0.01 * pk * sr
        obj = obj + e4 * e4

        # E6: CA^2 - 1.0 * EL2 (QD/SQ type)
        qd2 = ss - so - cb * lf
        sq2 = lf * lf
        el2 = qd2 / sq2
        e6 = ca - el2
        obj = obj + e6 * e6

        # E7: (-2*AM + SO + SS)^2 + 0.01 * EL3 (12: EF / LF)
        el3 = ef / lf
        e7 = -2.0 * am + so + ss + 0.01 * el3
        obj = obj + e7 * e7

        # E8: AM^2 - 0.25 * EL4 (12/1: SO * CB^2 / CA)
        el4 = so * cb * cb / ca
        e8 = am - 0.25 * el4
        obj = obj + e8 * e8

        # E9: IMPDER^2 - 1.3 * EL5 (SQ: SD^2) - 27.5 * SD
        el5 = sd * sd
        e9 = impder - 1.3 * el5 - 27.5 * sd
        obj = obj + e9 * e9

        # E10: IMPK^2 + 8.6 * EL6 (SQ: SK^2) - 70.0 * SK
        el6 = sk * sk
        e10 = impk + 8.6 * el6 - 70.0 * sk
        obj = obj + e10 * e10

        # E13: QI^2 + EL7/24000 (SQ: MV^2) - 1000.0
        el7 = mv * mv
        e13 = qi + el7 / 24000.0 - 1000.0
        obj = obj + e13 * e13

        # E14: PT * 1000^2 - EL8 (2PR: MD * PD)
        el8 = md * pd
        e14 = pt * 1000.0 - el8
        obj = obj + e14 * e14

        # E16: (VN + VS + QF/790)^2 - EL9 + EL10 - 2.0
        # EL9 (QT: MZ / CR), EL10 (2PR: DV * PT)
        el9 = mz / cr
        el10 = dv * pt
        e16 = vn + vs + qf / 790.0 - el9 + el10 - 2.0
        obj = obj + e16 * e16

        # E18: IMPMOT^2 - 1000 * EL11 - 12 * EL12
        # EL11 (1/LIN: PT / (PM + 20)), EL12 (SQRT: sqrt(PT))
        el11 = pt / (pm + 20.0)
        el12 = jnp.sqrt(pt)
        e18 = impmot - 1000.0 * el11 - 12.0 * el12
        obj = obj + e18 * e18

        # E26: ST^2 - 1.25 * EL13 (2PR: SR * NM)
        el13 = sr * nm
        e26 = st - 1.25 * el13
        obj = obj + e26 * e26

        # E27: SR^2 - EL14 (QT: MD / MS)
        el14 = md / ms
        e27 = sr - el14
        obj = obj + e27 * e27

        # E28: QV^2 - 2.4 * EL15 (SURD type)
        el15 = sx * jnp.sqrt(sx) * el / jnp.sqrt(lx)
        e28 = qv - 2.4 * el15
        obj = obj + e28 * e28

        # E29: SO^2 - 0.785 * EL16 (SQPRD: DE^2 * PT)
        el16 = de * de * pt
        e29 = so - 0.785 * el16
        obj = obj + e29 * e29

        # E30: SS^2 - 0.785 * EL17 (SQPRD: DS^2 * PT)
        el17 = ds * ds * pt
        e30 = ss - 0.785 * el17
        obj = obj + e30 * e30

        # E31: CB^2 - 2.0 * EL18 (CB/SQQD type)
        cb_num = vn - ca * lf * lf * lf
        sqqd = lf * lf * (3.0 - so * lf)
        el18 = cb_num / sqqd
        e31 = cb - 2.0 * el18
        obj = obj + e31 * e31

        # E32: IMPVOIL^2 - 1.15 * EL19 (SREL type)
        srlin = 15.0 + 0.15 * sx
        srpd = mc * lx
        srqd = 50.0 * sr * el
        srqt = srpd / srqd
        srrt = jnp.sqrt(srqt)
        el19 = sx * srlin * (srqt * srrt + 8.0)
        e32 = impvoil - 1.15 * el19
        obj = obj + e32 * e32

        return obj

    def constraint(self, y):
        """Compute the constraint functions."""
        # Extract variables
        sr, lr, pk, ef, sx, lx, sd, sk, st, sf = y[0:10]
        lf, am, ca, cb, so, ss, impder, impk, impfus = y[10:19]
        qi, pt, mv, mc, md, pd, ns, vs, cr, pm = y[19:29]
        dv, mz, vn, qv, qf, imptrain, impmot, impnmot = y[29:37]
        imppet, imppil, impcan, impsna, ms, el, de, ds = y[37:45]
        impvoil, nm, np, ng = y[45:49]

        # Equality constraints (all constraints are equalities in this problem)
        constraints = []

        # E1: SD - 0.13 * SR = 0
        constraints.append(sd - 0.13 * sr)

        # E2: SX - 0.7 * SR = 0
        constraints.append(sx - 0.7 * sr)

        # E3: LX - LR = 0
        constraints.append(lx - lr)

        # E5: SF - ST - 2*SD - 2*SX - 2*SK = 0
        constraints.append(sf - st - 2.0 * sd - 2.0 * sx - 2.0 * sk)

        # E11: IMPFUS - 20.0 * SF = 0
        constraints.append(impfus - 20.0 * sf)

        # E12: MD - 2.0 * MV = 0
        constraints.append(md - 2.0 * mv)

        # E15: QF - QI - QV = 0
        constraints.append(qf - qi - qv)

        # E17: IMPTRAIN - 0.137 * MV = 0
        constraints.append(imptrain - 0.137 * mv)

        # E19: IMPNMOT - 35.0 * NM = 0
        constraints.append(impnmot - 35.0 * nm)

        # E20: IMPPET - 0.043 * QI = 0
        constraints.append(imppet - 0.043 * qi)

        # E21: IMPPIL - 200.0 * NP = 0
        constraints.append(imppil - 200.0 * np)

        # E22: IMPCAN - 120.0 * NG = 0
        constraints.append(impcan - 120.0 * ng)

        # E23: IMPSNA - 300.0 * NS - 400.0 = 0
        constraints.append(impsna - 300.0 * ns - 400.0)

        # E24: MC - MV + 95*NP + 70*NG + 660*NM + 0.5*QI - 380.0 = 0
        constraints.append(
            mc - mv + 95.0 * np + 70.0 * ng + 660.0 * nm + 0.5 * qi - 380.0
        )

        # E25: MZ - IMPTRAIN + IMPNMOT + IMPPET + IMPPIL + IMPCAN + IMPSNA + 290.0 = 0
        constraints.append(
            mz - imptrain + impnmot + imppet + imppil + impcan + impsna + 290.0
        )

        # All constraints are equalities
        return jnp.array(constraints), None

    def y0(self):
        """Initial guess for variables."""
        return jnp.array(
            [
                2.7452e01,  # SR
                1.5000e00,  # LR
                1.0000e01,  # PK
                0.0000e00,  # EF
                1.9217e01,  # SX
                1.5000e00,  # LX
                3.5688e00,  # SD
                4.0696e00,  # SK
                3.4315e01,  # ST
                8.8025e01,  # SF
                5.1306e00,  # LF
                0.0000e00,  # AM
                -1.4809e-01,  # CA
                7.5980e-01,  # CB
                0.0000e00,  # SO
                0.0000e00,  # SS
                1.1470e02,  # IMPDER
                5.0000e02,  # IMPK
                1.7605e03,  # IMPFUS
                2.3256e03,  # QI
                5.6788e00,  # PT
                1.4197e04,  # MV
                1.2589e04,  # MC
                2.8394e04,  # MD
                2.0000e-01,  # PD
                1.0000e00,  # NS
                0.0000e00,  # VS
                1.0000e02,  # CR
                1.5000e01,  # PM
                0.0000e00,  # DV
                5.0000e02,  # MZ
                1.0000e01,  # VN
                8.1490e02,  # QV
                3.1405e03,  # QF
                1.9450e03,  # IMPTRAIN
                1.9085e02,  # IMPMOT
                3.5000e01,  # IMPNMOT
                1.0000e02,  # IMPPET
                2.0000e02,  # IMPPIL
                1.2000e02,  # IMPCAN
                7.0000e02,  # IMPSNA
                1.0000e03,  # MS
                4.9367e00,  # EL
                0.0000e00,  # DE
                0.0000e00,  # DS
                5.0000e03,  # IMPVOIL
                1.0000e00,  # NM
                1.0000e00,  # NP
                1.0000e00,  # NG
            ]
        )

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds."""
        lower = jnp.array(
            [
                10.0,  # SR
                0.0,  # LR
                0.0,  # PK
                0.0,  # EF
                7.0,  # SX
                1.5,  # LX
                2.0,  # SD
                2.0,  # SK
                30.0,  # ST
                20.0,  # SF
                0.001,  # LF
                0.0,  # AM
                -0.2,  # CA
                0.1,  # CB
                0.0,  # SO
                0.0,  # SS
                100.0,  # IMPDER
                500.0,  # IMPK
                500.0,  # IMPFUS
                1000.0,  # QI
                2.0,  # PT
                2000.0,  # MV
                3000.0,  # MC
                5000.0,  # MD
                0.2,  # PD
                1.0,  # NS
                0.0,  # VS
                100.0,  # CR
                4.0,  # PM
                0.0,  # DV
                500.0,  # MZ
                10.0,  # VN
                250.0,  # QV
                750.0,  # QF
                250.0,  # IMPTRAIN
                10.0,  # IMPMOT
                35.0,  # IMPNMOT
                100.0,  # IMPPET
                200.0,  # IMPPIL
                120.0,  # IMPCAN
                700.0,  # IMPSNA
                100.0,  # MS
                2.0,  # EL
                0.0,  # DE
                0.0,  # DS
                500.0,  # IMPVOIL
                1.0,  # NM
                1.0,  # NP
                1.0,  # NG
            ]
        )

        upper = jnp.array(
            [
                150.0,  # SR
                10.0,  # LR
                10.0,  # PK
                5.0,  # EF
                120.0,  # SX
                8.0,  # LX
                20.0,  # SD
                30.0,  # SK
                500.0,  # ST
                200.0,  # SF
                20.0,  # LF
                10.0,  # AM
                -0.001,  # CA
                2.0,  # CB
                1.0,  # SO
                2.0,  # SS
                1000.0,  # IMPDER
                5000.0,  # IMPK
                5000.0,  # IMPFUS
                20000.0,  # QI
                30.0,  # PT
                20000.0,  # MV
                30000.0,  # MC
                50000.0,  # MD
                0.8,  # PD
                5.0,  # NS
                20.0,  # VS
                400.0,  # CR
                15.0,  # PM
                10.0,  # DV
                10000.0,  # MZ
                50.0,  # VN
                5000.0,  # QV
                15000.0,  # QF
                3000.0,  # IMPTRAIN
                5000.0,  # IMPMOT
                70.0,  # IMPNMOT
                3000.0,  # IMPPET
                400.0,  # IMPPIL
                240.0,  # IMPCAN
                1900.0,  # IMPSNA
                1000.0,  # MS
                20.0,  # EL
                1.0,  # DE
                2.0,  # DS
                5000.0,  # IMPVOIL
                2.0,  # NM
                2.0,  # NP
                2.0,  # NG
            ]
        )

        return lower, upper

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From comment in SIF file
        return jnp.array(9.46801297093018e07)
