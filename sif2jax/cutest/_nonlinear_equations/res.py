"""Dassault France ressort (spring) problem.

SIF input: A. R. Conn, June 1993.

Classification: NLR2-MN-20-14
"""

import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class RES(AbstractNonlinearEquations):
    """Dassault France ressort (spring) problem with 20 variables and 14 equations."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Constants
    pi: float = 3.1415926535

    @property
    def n(self):
        """Number of variables."""
        return 20

    @property
    def m(self):
        """Number of residuals/equations."""
        return 14

    @property
    def y0(self):
        """Initial guess."""
        return jnp.array(
            [
                1.5000e-01,  # L0
                2.4079e01,  # N
                9.2459e-15,  # F
                0.0000e00,  # K
                0.0000e00,  # LB
                1.5000e-01,  # L
                6.8120e00,  # DE
                6.6120e00,  # DI
                0.0000e00,  # TO
                0.0000e00,  # TOB
                2.2079e01,  # NU
                1.0000e-01,  # D
                6.5268e-01,  # P
                5.5268e-01,  # E
                6.5887e02,  # P0
                6.5887e04,  # G
                6.7120e00,  # DM
                1.5000e-01,  # FR
                1.0000e02,  # TOLIM
                1.0000e02,  # TOBLIM
            ]
        )

    @property
    def args(self):
        """No additional arguments."""
        return None

    def residual(self, y, args):
        """Compute the residual vector."""
        del args  # Not used

        # Unpack variables
        L0 = y[0]
        N = y[1]
        F = y[2]
        K = y[3]
        LB = y[4]
        L = y[5]
        DE = y[6]
        DI = y[7]
        TO = y[8]
        TOB = y[9]
        NU = y[10]
        D = y[11]
        P = y[12]
        E = y[13]
        P0 = y[14]
        G = y[15]
        DM = y[16]
        FR = y[17]
        TOLIM = y[18]
        TOBLIM = y[19]

        # Compute element functions
        # EL1: 311/14 type with V=DM, W=NU, X=P0, Y=G, Z=D
        V3WX = DM**3 * NU * P0
        YZ4 = G * D**4
        EL1_val = V3WX / YZ4

        # EL2: 14/31 type with W=G, X=D, Y=DM, Z=NU
        WX4 = G * D**4
        Y3Z = DM**3 * NU
        EL2_val = WX4 / Y3Z

        # EL3: 2PR type with X=NU, Y=P
        EL3_val = NU * P

        # EL4: 11/3 type with X=P0, Y=DM, Z=D
        EL4_val = (P0 * DM) / (self.pi * D**3)

        # EL5: 111/2 type with W=G, X=D, Y=E, Z=DM
        EL5_val = (G * D * E) / (self.pi * DM**2)

        # Compute residuals
        res = jnp.zeros(14)

        # E1: F = EL1
        res = res.at[0].set(F - EL1_val)

        # E2: K = EL2
        res = res.at[1].set(K - EL2_val)

        # E3: DE - D - DM = 0
        res = res.at[2].set(DE - D - DM)

        # E4: DI + D - DM = 0
        res = res.at[3].set(DI + D - DM)

        # E5: D - P + E = 0
        res = res.at[4].set(D - P + E)

        # E6: NU - N = -2.0
        res = res.at[5].set(NU - N + 2.0)

        # E7: 1.5*D - L0 = 0
        res = res.at[6].set(1.5 * D - L0)

        # E8: L - LB - FR = 0
        res = res.at[7].set(L - LB - FR)

        # E9: LB = EL3
        res = res.at[8].set(LB - EL3_val)

        # E10: L - L0 + F = EL4
        res = res.at[9].set(L - L0 + F - EL4_val)

        # E11: TO = EL5
        res = res.at[10].set(TO - EL5_val)

        # E12: TOB = EL5
        res = res.at[11].set(TOB - EL5_val)

        # E13: TO - TOLIM <= 0 (inequality)
        res = res.at[12].set(jnp.maximum(0.0, TO - TOLIM))

        # E14: TOB - TOBLIM <= 0 (inequality)
        res = res.at[13].set(jnp.maximum(0.0, TOB - TOBLIM))

        return res

    def constraint(self, y):
        """Return the constraint values as required by the abstract base class."""
        # For nonlinear equations, residuals are equality constraints
        residuals = self.residual(y, self.args)

        # Separate equality and inequality constraints
        # E1-E12 are equalities (indices 0-11)
        # E13-E14 are inequalities (indices 12-13)
        equality_constraints = residuals[:12]
        inequality_constraints = residuals[12:]

        return equality_constraints, inequality_constraints

    @property
    def bounds(self):
        """Bounds on variables."""
        lower = jnp.array(
            [
                0.0,  # L0
                0.0,  # N
                0.0,  # F
                0.0,  # K
                0.0,  # LB
                0.0,  # L
                0.0,  # DE
                0.0,  # DI
                0.0,  # TO
                0.0,  # TOB
                0.5,  # NU
                0.1,  # D
                0.0,  # P
                0.0,  # E
                1.0,  # P0
                40000.0,  # G
                0.1,  # DM
                0.0,  # FR
                100.0,  # TOLIM
                100.0,  # TOBLIM
            ]
        )

        upper = jnp.array(
            [
                100.0,  # L0
                100.0,  # N
                30.0,  # F
                100.0,  # K
                50.0,  # LB
                50.0,  # L
                30.0,  # DE
                30.0,  # DI
                800.0,  # TO
                800.0,  # TOB
                50.0,  # NU
                10.0,  # D
                20.0,  # P
                10.0,  # E
                1000.0,  # P0
                80000.0,  # G
                30.0,  # DM
                50.0,  # FR
                1000.0,  # TOLIM
                1000.0,  # TOBLIM
            ]
        )

        return (lower, upper)

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value is 0.0 for nonlinear equations."""
        return jnp.array(0.0)
