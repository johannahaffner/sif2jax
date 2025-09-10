import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class HS99EXP(AbstractConstrainedMinimisation):
    """Hock and Schittkowski problem 99 expanded form.

    Source: an expanded form of problem 99 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: Ph. Toint, April 1991.

    classification OOR2-AN-31-21
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        # 7 X variables + 8 R variables + 8 Q variables + 8 S variables = 31
        # R(1), Q(1), S(1) are fixed at 0 via bounds
        return 31

    @property
    def y0(self):
        """Initial guess."""
        # From START POINT: X(i) = 0.5 for i=1..7, others default to 0
        # Variables are interleaved: X(1), R(2), Q(2), S(2), X(2), R(3), Q(3), S(3), ...
        y0 = jnp.zeros(31, dtype=jnp.float64)
        # X(1) at position 0
        y0 = y0.at[0].set(0.5)
        # X(2:7) at positions 4, 8, 12, 16, 20, 24
        positions = jnp.array([4, 8, 12, 16, 20, 24])
        y0 = y0.at[positions].set(0.5)
        # Note: R(1), Q(1), S(1) don't appear as variables
        return y0

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Objective function."""
        del args
        # Variables are interleaved: X(1), R(2), Q(2), S(2), X(2), R(3), Q(3), S(3), ...
        # ..., X(7), R(8), Q(8), S(8), R(1), Q(1), S(1)
        # R(8) is at position 25
        r8 = y[25]

        # Objective: -R(8)^2 (with L2 group type and SCALE -1.0)
        return -r8 * r8

    @property
    def bounds(self):
        """Variable bounds."""
        # FR (free) is default for all variables
        lower = jnp.full(31, -jnp.inf, dtype=jnp.float64)
        upper = jnp.full(31, jnp.inf, dtype=jnp.float64)

        # Variables are interleaved: X(1), R(2), Q(2), S(2), X(2), R(3), Q(3), S(3), ...
        # ..., X(7), R(8), Q(8), S(8), R(1), Q(1), S(1)

        # X(1:7) have bounds [0, 1.58]
        # X(1) at position 0, X(2:7) at positions 4, 8, 12, 16, 20, 24
        lower = lower.at[0].set(0.0)
        upper = upper.at[0].set(1.58)
        positions = jnp.array([4, 8, 12, 16, 20, 24])  # X(2:7)
        lower = lower.at[positions].set(0.0)
        upper = upper.at[positions].set(1.58)

        # R(1), Q(1), S(1) are fixed at 0 (positions 28, 29, 30)
        lower = lower.at[28].set(0.0)  # R(1)
        upper = upper.at[28].set(0.0)  # R(1)
        lower = lower.at[29].set(0.0)  # Q(1)
        upper = upper.at[29].set(0.0)  # Q(1)
        lower = lower.at[30].set(0.0)  # S(1)
        upper = upper.at[30].set(0.0)  # S(1)

        return lower, upper

    def constraint(self, y):
        """Constraint functions."""
        # Variables are interleaved: X(1), R(2), Q(2), S(2), X(2), R(3), Q(3), S(3), ...
        # ..., X(7), R(8), Q(8), S(8), R(1), Q(1), S(1)

        # Extract variables
        x = jnp.zeros(7)  # X(1:7)
        r = jnp.zeros(8)  # R(1:8)
        q = jnp.zeros(8)  # Q(1:8)
        s = jnp.zeros(8)  # S(1:8)

        # X(1) at position 0
        x = x.at[0].set(y[0])

        # Pattern: after X(1), we have groups of [R(i), Q(i), S(i), X(i)] for i=2..7
        # Vectorized extraction
        i_range = jnp.arange(2, 8)
        base_positions = 1 + 4 * (i_range - 2)  # Start positions for groups

        r = r.at[i_range - 1].set(y[base_positions])  # R(2:7)
        q = q.at[i_range - 1].set(y[base_positions + 1])  # Q(2:7)
        s = s.at[i_range - 1].set(y[base_positions + 2])  # S(2:7)
        x = x.at[i_range - 1].set(y[base_positions + 3])  # X(2:7)

        # R(8), Q(8), S(8) at positions 25, 26, 27
        r = r.at[7].set(y[25])
        q = q.at[7].set(y[26])
        s = s.at[7].set(y[27])

        # R(1), Q(1), S(1) at positions 28, 29, 30 (fixed at 0)
        r = r.at[0].set(y[28])
        q = q.at[0].set(y[29])
        s = s.at[0].set(y[30])

        # Time parameters from SIF file
        t = jnp.array([0.0, 25.0, 50.0, 100.0, 150.0, 200.0, 290.0, 380.0])
        dt = t[1:] - t[:-1]  # Time increments

        # A parameters from SIF file
        a = jnp.array([0.0, 50.0, 50.0, 75.0, 75.0, 75.0, 100.0, 100.0])

        # B parameter
        B = 32.0

        # Trigonometric values
        cos_x = jnp.cos(x)
        sin_x = jnp.sin(x)

        # Constraints: 21 equality constraints
        eq_constraints = []

        # R(i)DEF constraints: R(i) - R(i-1) - A(i)*DT(i)*cos(X(i-1)) = 0 for i=2..8
        # Vectorized computation for i=1..7 (representing I=2..8 in SIF)
        i_indices = jnp.arange(1, 8)
        r_constraints = (
            r[i_indices]
            - r[i_indices - 1]
            - a[i_indices] * dt[i_indices - 1] * cos_x[i_indices - 1]
        )
        eq_constraints.extend(r_constraints)

        # Q(i)DEF constraints: Q(i) - Q(i-1) - DT(i)*S(i-1) - A(i)*DT(i)^2/2*sin(X(i-1))
        # = RHS for i=2..8
        # RHS = B*DT(i)^2/2 for i=2..7, special values for i=8
        # Vectorized computation
        rhs_q = jnp.where(i_indices < 7, B * dt[i_indices - 1] ** 2 / 2.0, 100000.0)
        q_constraints = (
            q[i_indices]
            - q[i_indices - 1]
            - dt[i_indices - 1] * s[i_indices - 1]
            - a[i_indices] * dt[i_indices - 1] ** 2 / 2.0 * sin_x[i_indices - 1]
            - rhs_q
        )
        eq_constraints.extend(q_constraints)

        # S(i)DEF constraints: S(i) - S(i-1) - A(i)*DT(i)*sin(X(i-1)) = RHS for i=2..8
        # RHS = B*DT(i) for i=2..7, special value for i=8
        # Vectorized computation
        rhs_s = jnp.where(i_indices < 7, B * dt[i_indices - 1], 1000.0)
        s_constraints = (
            s[i_indices]
            - s[i_indices - 1]
            - a[i_indices] * dt[i_indices - 1] * sin_x[i_indices - 1]
            - rhs_s
        )
        eq_constraints.extend(s_constraints)

        equalities = jnp.array(eq_constraints)

        return equalities, None

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided in SIF file."""
        return None
