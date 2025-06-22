import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO needs human review, Claude tried and failed (and I don't know what was tried)
class DALLASS(AbstractConstrainedMinimisation):
    """The small Dallas water distribution problem.

    The problem is also named "W30" in some references.
    This is a nonlinear network problem with conditioning of
    the order of 10**4.

    Source:
    R. Dembo,
    private communication, 1986.

    SIF input: Ph. Toint, June 1990.

    Classification: ONR2-MN-46-31
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 46

    @property
    def m(self):
        """Number of constraints."""
        return 31

    def objective(self, y, args):
        """Compute the objective function.

        The objective consists of nonlinear pipe flow elements and linear terms.
        """
        raise NotImplementedError("DALLASS requires complex element implementations")

    def equality_constraints(self, y, args):
        """Compute the equality constraints.

        These are network flow conservation constraints.
        """
        del args
        x = y

        constraints = []

        # N1: X46 + X41 - X1 = 0
        constraints.append(x[45] + x[40] - x[0])
        # N2: X45 - X2 = 0
        constraints.append(x[44] - x[1])
        # N3: X44 - X3 = 0
        constraints.append(x[43] - x[2])
        # N4: -X4 = 0
        constraints.append(-x[3])
        # N5: X16 + X7 - X6 - X5 = 2.8
        constraints.append(x[15] + x[6] - x[5] - x[4] - 2.8)
        # N6: X7 - X9 - X8 = 0
        constraints.append(x[6] - x[8] - x[7])
        # N7: X9 - X10 = 0.403
        constraints.append(x[8] - x[9] - 0.403)
        # N8: X10 + X2 - X12 - X11 = 0.592
        constraints.append(x[9] + x[1] - x[11] - x[10] - 0.592)
        # N9: X12 + X3 - X13 = 1.156
        constraints.append(x[11] + x[2] - x[12] - 1.156)
        # N10: X4 - X16 - X15 - X14 = 0.2
        constraints.append(x[3] - x[15] - x[14] - x[13] - 0.2)
        # N11: X15 + X13 + X5 - X17 = 0.495
        constraints.append(x[14] + x[12] + x[4] - x[16] - 0.495)
        # N12: X20 + X19 + X8 + X6 - X18 = 0
        constraints.append(x[19] + x[18] + x[7] + x[5] - x[17])
        # N13: X42 + X18 - X19 = 0
        constraints.append(x[41] + x[17] - x[18])
        # N14: X21 - X20 = 0
        constraints.append(x[20] - x[19])
        # N15: X43 - X21 = 0
        constraints.append(x[42] - x[20])
        # N16: X14 + X11 - X23 - X22 = 0.313
        constraints.append(x[13] + x[10] - x[22] - x[21] - 0.313)
        # N17: X23 - X25 - X24 = 0.844
        constraints.append(x[22] - x[24] - x[23] - 0.844)
        # N18: X31 + X25 + X22 - X26 = 0.331
        constraints.append(x[30] + x[24] + x[21] - x[25] - 0.331)
        # N19: X26 + X17 - X28 - X27 = 0.053
        constraints.append(x[25] + x[16] - x[27] - x[26] - 0.053)
        # N20: X28 = 0
        constraints.append(x[27])
        # N21: -X31 - X30 - X29 = 0.272
        constraints.append(-x[30] - x[29] - x[28] - 0.272)
        # N22: X30 + X27 = 0.883
        constraints.append(x[29] + x[26] - 0.883)
        # N23: X24 - X32 = 0.571
        constraints.append(x[23] - x[31] - 0.571)
        # N24: X38 + X29 - X34 - X33 = 0.755
        constraints.append(x[37] + x[28] - x[33] - x[32] - 0.755)
        # N25: X32 - X35 = 0
        constraints.append(x[31] - x[34])
        # N26: X35 - X37 - X36 = 0.527
        constraints.append(x[34] - x[36] - x[35] - 0.527)
        # N27: X37 + X34 = 0
        constraints.append(x[36] + x[33])
        # N28: X36 - X40 - X39 - X38 = 0
        constraints.append(x[35] - x[39] - x[38] - x[37])
        # N29: X39 + X33 + X1 = 0.001
        constraints.append(x[38] + x[32] + x[0] - 0.001)
        # N30: X40 - X41 = 0
        constraints.append(x[39] - x[40])
        # N31: -X46 - X45 - X44 - X43 - X42 = -10.196
        constraints.append(-x[45] - x[44] - x[43] - x[42] - x[41] + 10.196)

        return jnp.array(constraints), None

    def bounds(self):
        """Variable bounds."""
        lower = jnp.full(self.n, -200.0)
        upper = jnp.full(self.n, 200.0)

        # Special bounds
        lower = lower.at[0].set(0.0)  # X1
        upper = upper.at[0].set(21.1673)
        lower = lower.at[1].set(0.0)  # X2
        upper = upper.at[1].set(43.7635)
        lower = lower.at[2].set(0.0)  # X3
        upper = upper.at[2].set(32.8255)
        lower = lower.at[18].set(0.0)  # X19
        upper = upper.at[18].set(22.0120)
        lower = lower.at[20].set(0.0)  # X21
        upper = upper.at[20].set(13.6703)

        return lower, upper

    def y0(self):
        """Initial guess."""
        y0 = jnp.full(self.n, -200.0)

        # Set specific initial values
        y0 = y0.at[0].set(21.1673)  # X1
        y0 = y0.at[1].set(43.7635)  # X2
        y0 = y0.at[2].set(32.8255)  # X3
        y0 = y0.at[3].set(1.42109e-14)  # X4
        y0 = y0.at[4].set(168.826)  # X5
        y0 = y0.at[6].set(28.1745)  # X7
        y0 = y0.at[7].set(87.5603)  # X8
        y0 = y0.at[8].set(-59.3858)  # X9
        y0 = y0.at[9].set(-59.7888)  # X10
        y0 = y0.at[10].set(183.383)  # X11
        y0 = y0.at[12].set(-168.331)  # X13
        y0 = y0.at[14].set(200.0)  # X15
        y0 = y0.at[15].set(0.2)  # X16
        y0 = y0.at[16].set(200.0)  # X17
        y0 = y0.at[17].set(-76.7574)  # X18
        y0 = y0.at[18].set(22.0120)  # X19
        y0 = y0.at[19].set(13.6703)  # X20
        y0 = y0.at[20].set(13.6703)  # X21
        y0 = y0.at[21].set(-198.461)  # X22
        y0 = y0.at[22].set(181.531)  # X23
        y0 = y0.at[23].set(-19.3133)  # X24
        y0 = y0.at[24].set(200.0)  # X25
        y0 = y0.at[25].set(-198.792)  # X26
        y0 = y0.at[26].set(1.155)  # X27
        y0 = y0.at[27].set(0.0)  # X28
        y0 = y0.at[28].set(200.0)  # X29
        y0 = y0.at[29].set(0.272)  # X30
        y0 = y0.at[31].set(-19.8843)  # X32
        y0 = y0.at[32].set(178.834)  # X33
        y0 = y0.at[33].set(-179.589)  # X34
        y0 = y0.at[34].set(-19.8843)  # X35
        y0 = y0.at[36].set(179.589)  # X37
        y0 = y0.at[39].set(200.0)  # X40
        y0 = y0.at[40].set(200.0)  # X41
        y0 = y0.at[41].set(98.7694)  # X42
        y0 = y0.at[42].set(13.6703)  # X43
        y0 = y0.at[43].set(32.8255)  # X44
        y0 = y0.at[44].set(43.7635)  # X45
        y0 = y0.at[45].set(-178.833)  # X46

        return y0

    def args(self):
        """No additional arguments."""
        return None

    def expected_result(self):
        """Expected optimal solution."""
        # Not provided in SIF file
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file comment
        return jnp.array(-32393.0)
