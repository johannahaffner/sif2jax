import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed - problem structure doesn't match pycutest
# The objective/constraint split needs to be reconsidered
# TODO: Human review needed - ANTWERP
# Attempts made:
# 1. Fixed constraint formulation (AGE2/AGE3 now correctly use N2/N3)
# 2. Initial value calculation follows SIF file formulas
# Suspected issues: Initial probability calculations may differ from pycutest defaults
# Additional resources needed: Understanding of pycutest initial value handling
class ANTWERP(AbstractConstrainedMinimisation):
    """Synthetic population estimation problem for Antwerp.

    This problem estimates the distribution of households in Belgian municipalities.
    Three household types are considered:
    - Type F: a couple + 1 to 5 children + 0 to 2 additional adults
    - Type W: a woman + 1 to 3 children + 0 to 2 additional adults
    - Type M: a man + 1 to 3 children + 0 to 2 additional adults

    The problem is noted to be very ill-conditioned.

    Source: L. Schoonbeek, "A household disaggregation procedure for the analysis
    of labour market policies", presented at the Franco-Belgian Symposium on
    "Desaggregation Methods in Socio-Economics", Brussels, 1986.

    SIF input: Nick Gould, February 1991

    Classification: SLR2-RN-27-8-0-3-24-0-2-0-8-0-0-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem data - demographic statistics for Antwerp
    # Number of households by size
    m3: float = 5827.0
    m4: float = 5929.0
    m5: float = 2889.0
    m6: float = 1065.0
    m7: float = 364.0
    m8: float = 180.0

    # Number of households by type and children
    m1f: float = 1974.0
    m2f: float = 5645.0
    m3f: float = 2813.0

    m1w: float = 2967.0
    m2w: float = 1095.0
    m3w: float = 296.0

    m1m: float = 3023.0
    m2m: float = 1133.0
    m3m: float = 327.0

    # Population counts
    n0: float = 27435.0  # Children (by type)
    n1: float = 3359.0  # Adults in W households
    n2: float = 4444.0  # Adults in M households
    n3: float = 14838.0  # Adults in F households
    n4: float = 18949.0  # Adults in F households (different age group)

    ninf: float = 48262.0  # Total individuals in F households
    ninn: float = 17784.0  # Total individuals in N households

    def __init__(self):
        """Initialize the ANTWERP problem."""
        # No fields to initialize as all data is defined at class level
        pass

    @property
    def n(self):
        """Total number of variables."""
        return 27

    @property
    def m(self):
        """Total number of constraints."""
        return 10  # 8 equality + 2 inequality

    def objective(self, y, args):
        """Compute the least-squares objective function."""
        del args

        # Extract variables
        p1f, p2f, p3f, p4f, p5f = y[0:5]
        p1w, p2w, p3w = y[5:8]
        p1m, p2m, p3m = y[8:11]
        q0f, q1f, q2f = y[11:14]
        q0w, q1w, q2w = y[14:17]
        q0m, q1m, q2m = y[17:20]
        nf, nw, nm = y[20:23]
        nc2, na2, nc3, na3 = y[23:27]

        # Compute household size predictions
        hsz3 = (p1f * q0f + p1f * q1f) * nf + p1w * q0w * nw + p1m * q0m * nm
        hsz4 = (
            (p2f * q0f + p1f * q2f + p2f * q1f) * nf
            + (p1w * q1w + p2w * q0w) * nw
            + (p1m * q1m + p2m * q0m) * nm
        )
        hsz5 = (
            (p3f * q0f + p2f * q2f + p3f * q1f) * nf
            + (p1w * q2w + p2w * q1w + p3w * q0w) * nw
            + (p1m * q2m + p2m * q1m + p3m * q0m) * nm
        )
        hsz6 = (
            (p4f * q0f + p3f * q2f + p4f * q1f) * nf
            + (p2w * q2w + p3w * q1w) * nw
            + (p2m * q2m + p3m * q1m) * nm
        )
        hsz7 = (
            (p5f * q0f + p4f * q2f + p5f * q1f) * nf + p3w * q2w * nw + p3m * q2m * nm
        )
        hsz8 = (p5f * q2f) * nf

        # Compute household type predictions
        hst1f = p1f * nf
        hst2f = p2f * nf
        hst3f = (p3f + p4f + p5f) * nf
        hst1w = p1w * nw
        hst2w = p2w * nw
        hst3w = p3w * nw
        hst1m = p1m * nm
        hst2m = p2m * nm
        hst3m = p3m * nm

        # Compute population predictions
        hch = (
            (p1f + 2 * p2f + 3 * p3f + 4 * p4f + 5 * p5f) * nf
            + (p1w + 2 * p2w + 3 * p3w) * nw
            + (p1m + 2 * p2m + 3 * p3m) * nm
            + nc2
            + nc3
        )
        had = (
            2 * nf
            + (q1f + 2 * q2f) * nf
            + (q1w + 2 * q2w) * nw
            + (q1m + 2 * q2m) * nm
            + na2
            + na3
        )
        hinf = nf * (3 + p1f + 2 * p2f + 3 * p3f + 4 * p4f + 5 * p5f + q1f + 2 * q2f)
        hinn = (
            nw * (1 + p1w + 2 * p2w + 3 * p3w + q1w + 2 * q2w)
            + nm * (1 + p1m + 2 * p2m + 3 * p3m + q1m + 2 * q2m)
            + nc2
            + na2
            + nc3
            + na3
        )

        # Compute squared errors (scaled by inverse of RHS)
        obj = 0.0
        obj += ((hsz3 - self.m3) / self.m3) ** 2
        obj += ((hsz4 - self.m4) / self.m4) ** 2
        obj += ((hsz5 - self.m5) / self.m5) ** 2
        obj += ((hsz6 - self.m6) / self.m6) ** 2
        obj += ((hsz7 - self.m7) / self.m7) ** 2
        obj += ((hsz8 - self.m8) / self.m8) ** 2

        obj += ((hst1f - self.m1f) / self.m1f) ** 2
        obj += ((hst2f - self.m2f) / self.m2f) ** 2
        obj += ((hst3f - self.m3f) / self.m3f) ** 2
        obj += ((hst1w - self.m1w) / self.m1w) ** 2
        obj += ((hst2w - self.m2w) / self.m2w) ** 2
        obj += ((hst3w - self.m3w) / self.m3w) ** 2
        obj += ((hst1m - self.m1m) / self.m1m) ** 2
        obj += ((hst2m - self.m2m) / self.m2m) ** 2
        obj += ((hst3m - self.m3m) / self.m3m) ** 2

        obj += ((hch - self.n0) / self.n0) ** 2
        nad_total = self.n1 + self.n2 + self.n3 + self.n4
        obj += ((had - nad_total) / nad_total) ** 2
        obj += ((hinf - self.ninf) / self.ninf) ** 2
        obj += ((hinn - self.ninn) / self.ninn) ** 2

        return obj

    def constraint(self, y):
        """Compute equality and inequality constraints."""
        # Extract variables
        p1f, p2f, p3f, p4f, p5f = y[0:5]
        p1w, p2w, p3w = y[5:8]
        p1m, p2m, p3m = y[8:11]
        q0f, q1f, q2f = y[11:14]
        q0w, q1w, q2w = y[14:17]
        q0m, q1m, q2m = y[17:20]
        nf, nw, nm = y[20:23]
        nc2, na2, nc3, na3 = y[23:27]

        # Equality constraints
        eq_constraints = []

        # AGE2: NC2 + NA2 = N2
        eq_constraints.append(nc2 + na2 - self.n2)

        # AGE3: NC3 + NA3 = N3
        eq_constraints.append(nc3 + na3 - self.n3)

        # PSF: P1F + P2F + P3F + P4F + P5F = 1
        eq_constraints.append(p1f + p2f + p3f + p4f + p5f - 1.0)

        # PSW: P1W + P2W + P3W = 1
        eq_constraints.append(p1w + p2w + p3w - 1.0)

        # PSM: P1M + P2M + P3M = 1
        eq_constraints.append(p1m + p2m + p3m - 1.0)

        # QSF: Q0F + Q1F + Q2F = 1
        eq_constraints.append(q0f + q1f + q2f - 1.0)

        # QSW: Q0W + Q1W + Q2W = 1
        eq_constraints.append(q0w + q1w + q2w - 1.0)

        # QSM: Q0M + Q1M + Q2M = 1
        eq_constraints.append(q0m + q1m + q2m - 1.0)

        # Inequality constraints (>= 0)
        ineq_constraints = []

        # INEQ2: NA2 - 2642 >= 0
        ineq_constraints.append(na2 - 2642.0)

        # INEQ3: NA3 >= 0
        ineq_constraints.append(na3)

        return jnp.array(eq_constraints), jnp.array(ineq_constraints)

    def y0(self):
        """Initial guess for variables."""
        # Compute initial values based on data
        sp1f = self.m1f / (self.m1f + self.m2f + self.m3f)
        sp2f = self.m2f / (self.m1f + self.m2f + self.m3f)
        sp1w = self.m1w / (self.m1w + self.m2w + self.m3w)
        sp2w = self.m2w / (self.m1w + self.m2w + self.m3w)
        sp3w = self.m3w / (self.m1w + self.m2w + self.m3w)
        sp1m = self.m1m / (self.m1m + self.m2m + self.m3m)
        sp2m = self.m2m / (self.m1m + self.m2m + self.m3m)
        sp3m = self.m3m / (self.m1m + self.m2m + self.m3m)

        snf = self.m1f + self.m2f + self.m3f
        snw = self.m1w + self.m2w + self.m3w
        snm = self.m1m + self.m2m + self.m3m

        return jnp.array(
            [
                sp1f,  # P1F
                sp2f,  # P2F
                0.15,  # P3F
                0.10,  # P4F
                0.05,  # P5F
                sp1w,  # P1W
                sp2w,  # P2W
                sp3w,  # P3W
                sp1m,  # P1M
                sp2m,  # P2M
                sp3m,  # P3M
                0.6,  # Q0F
                0.3,  # Q1F
                0.1,  # Q2F
                0.6,  # Q0W
                0.3,  # Q1W
                0.1,  # Q2W
                0.6,  # Q0M
                0.3,  # Q1M
                0.1,  # Q2M
                snf,  # NF
                snw,  # NW
                snm,  # NM
                0.0,  # NC2
                self.n2,  # NA2
                0.0,  # NC3
                self.n3,  # NA3
            ]
        )

    def bounds(self):
        """Get variable bounds."""
        # All probabilities in [0, 1], except P3F, P4F, P5F with tighter bounds
        # NF, NW, NM in [0, 10000]
        # NC2, NC3 are fixed at 0
        # NA2, NA3 are free
        lower = jnp.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,  # P1F-P5F
                0.0,
                0.0,
                0.0,  # P1W-P3W
                0.0,
                0.0,
                0.0,  # P1M-P3M
                0.0,
                0.0,
                0.0,  # Q0F-Q2F
                0.0,
                0.0,
                0.0,  # Q0W-Q2W
                0.0,
                0.0,
                0.0,  # Q0M-Q2M
                0.0,
                0.0,
                0.0,  # NF, NW, NM
                0.0,
                -jnp.inf,  # NC2, NA2
                0.0,
                -jnp.inf,  # NC3, NA3
            ]
        )

        upper = jnp.array(
            [
                1.0,
                1.0,
                0.25,
                0.2,
                0.1,  # P1F-P5F
                1.0,
                1.0,
                1.0,  # P1W-P3W
                1.0,
                1.0,
                1.0,  # P1M-P3M
                1.0,
                1.0,
                1.0,  # Q0F-Q2F
                1.0,
                1.0,
                1.0,  # Q0W-Q2W
                1.0,
                1.0,
                1.0,  # Q0M-Q2M
                10000.0,
                10000.0,
                10000.0,  # NF, NW, NM
                0.0,
                jnp.inf,  # NC2, NA2
                0.0,
                jnp.inf,  # NC3, NA3
            ]
        )

        return (lower, upper)

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None

    def equality_constraints(self):
        """Mark which constraints are equalities."""
        # First 8 are equalities, last 2 are inequalities
        return jnp.array([True, True, True, True, True, True, True, True, False, False])
