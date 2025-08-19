import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class NASH(AbstractConstrainedQuadraticProblem):
    """A quadratic programming reformulation of a linear complementarity problem.

    A quadratic programming reformulation of a linear
    complementarity problem arising from Nash equilibrium
    provided by Michael Ferris

    SIF input: Michael Ferris, July 1993.

    classification QLR2-AN-72-24
    """

    n: int = 72
    m: int = 24
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Initial guess - all zeros except for free variables."""
        x0 = jnp.zeros(self.n)
        # Variables 1, 8-24 are free (non-zero initial values not specified in SIF)
        return x0

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Quadratic objective function.

        The objective has:
        - Linear terms from X56, X57, X58
        - Quadratic terms from the QUADRATIC section
        """
        del args

        # Linear objective terms (from GROUPS section)
        linear_obj = 1000.0 * y[55] + 500.0 * y[56] + 1000.0 * y[57]

        # Quadratic terms: 0.5 * x^T Q x
        # From QUADRATIC section, we have bilinear terms between X25-X48 and X1-X24
        # and between X49-X72 and X1-X24
        quad_obj = 0.0

        # X25-X48 paired with X1-X24 (coefficient 1.0)
        for i in range(24):
            quad_obj += y[24 + i] * y[i]

        # X49-X72 paired with X1-X24 (coefficient -1.0)
        for i in range(24):
            quad_obj += -y[48 + i] * y[i]

        return linear_obj + quad_obj

    @property
    def bounds(self):
        """Variable bounds.

        Most variables default to 0 (fixed), with exceptions:
        - X1: free
        - X8: [0, 1000]
        - X9: [0, 500]
        - X10: [0, 1000]
        - X11-X24: free
        """
        lower = jnp.zeros(self.n)
        upper = jnp.zeros(self.n)

        # Set free variables (represented as -inf/inf)
        free_vars = [0] + list(range(10, 24))  # X1, X11-X24 (0-indexed)
        for i in free_vars:
            lower = lower.at[i].set(-jnp.inf)
            upper = upper.at[i].set(jnp.inf)

        # Set bounded variables
        upper = upper.at[7].set(1000.0)  # X8
        upper = upper.at[8].set(500.0)  # X9
        upper = upper.at[9].set(1000.0)  # X10

        return lower, upper

    def constraint(self, y):
        """Linear equality constraints C1-C24.

        Returns equality constraints only (no inequalities).
        """
        c = jnp.zeros(24)

        # C1
        c = c.at[0].set(-y[24] + y[48] + y[0] - y[1] - y[2] - y[3] - y[4] - y[5] - y[6])

        # C2
        c = c.at[1].set(
            -y[25]
            + y[49]
            + 0.02309 * (y[0] + y[1] + y[3] + y[5])
            + y[10]
            + 0.288626 * y[21]
            + 0.263887 * y[22]
            + 0.447486 * y[23]
            - 35.100673
        )

        # C3
        c = c.at[2].set(
            -y[26]
            + y[50]
            + 0.02309 * (y[0] + y[2] + y[4] + y[6])
            + y[11]
            + 0.288626 * y[21]
            + 0.263887 * y[22]
            + 0.447486 * y[23]
            - 35.100673
        )

        # C4
        c = c.at[3].set(
            -y[27]
            + y[51]
            + 0.02309 * (y[0] + y[1] + y[3] + y[5])
            + y[10]
            + 0.288626 * y[21]
            + 0.263887 * y[22]
            + 0.447486 * y[23]
            - 35.100673
        )

        # C5
        c = c.at[4].set(
            -y[28]
            + y[52]
            + 0.02309 * (y[0] + y[2] + y[4] + y[6])
            + y[11]
            + 0.288626 * y[21]
            + 0.263887 * y[22]
            + 0.447486 * y[23]
            - 35.100673
        )

        # C6
        c = c.at[5].set(
            -y[29]
            + y[53]
            + 0.02309 * (y[0] + y[1] + y[3] + y[5])
            + y[10]
            + 0.288626 * y[21]
            + 0.263887 * y[22]
            + 0.447486 * y[23]
            - 35.100673
        )

        # C7
        c = c.at[6].set(
            -y[30]
            + y[54]
            + 0.02309 * (y[0] + y[2] + y[4] + y[6])
            + y[11]
            + 0.288626 * y[21]
            + 0.263887 * y[22]
            + 0.447486 * y[23]
            - 35.100673
        )

        # C8
        c = c.at[7].set(-y[31] + y[55] - y[10] - y[21] + 15.0)

        # C9
        c = c.at[8].set(-y[32] + y[56] - y[10] - y[22] + 15.0)

        # C10
        c = c.at[9].set(-y[33] + y[57] - y[11] - y[22] + 20.0)

        # C11
        c = c.at[10].set(-y[34] + y[58] - y[1] - y[3] - y[5] + y[7] + y[8])

        # C12
        c = c.at[11].set(-y[35] + y[59] - y[2] - y[4] - y[6] + y[9])

        # C13
        c = c.at[12].set(-y[36] + y[60] + y[15] - 0.33 * y[18] + 0.33 * y[19])

        # C14
        c = c.at[13].set(-y[37] + y[61] + y[16] - 0.67 * y[18] - 0.33 * y[19])

        # C15
        c = c.at[14].set(-y[38] + y[62] + y[17] - 0.33 * y[18] - 0.67 * y[19])

        # C16
        c = c.at[15].set(-y[39] + y[63] - y[12])

        # C17
        c = c.at[16].set(-y[40] + y[64] - y[13])

        # C18
        c = c.at[17].set(-y[41] + y[65] - y[14])

        # C19
        c = c.at[18].set(
            -y[42] + y[66] + 0.33 * y[12] + 0.67 * y[13] + 0.33 * y[14] - y[21]
        )

        # C20
        c = c.at[19].set(
            -y[43] + y[67] - 0.33 * y[12] + 0.33 * y[13] + 0.67 * y[14] - y[22]
        )

        # C21
        c = c.at[20].set(-y[44] + y[68] - y[23])

        # C22
        c = c.at[21].set(
            -y[45]
            + y[69]
            - 0.288626 * y[0]
            + y[7]
            + y[18]
            + 8.892169 * y[21]
            - 3.298588 * y[22]
            - 5.593581 * y[23]
            - 61.241589
        )

        # C23
        c = c.at[22].set(
            -y[46]
            + y[70]
            - 0.263887 * y[0]
            + y[8]
            + y[9]
            + y[19]
            - 3.298588 * y[21]
            + 8.412719 * y[22]
            - 5.114131 * y[23]
            + 1.150548
        )

        # C24
        c = c.at[23].set(
            -y[47]
            + y[71]
            - 0.447486 * y[0]
            + y[20]
            - 5.593581 * y[21]
            - 5.114131 * y[22]
            + 10.707712 * y[23]
            + 60.091041
        )

        return c, None  # Only equality constraints

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided."""
        return None
