"""
TRAINH: Optimal Train Energy Problem (Hilly Track).

TODO: Human review needed
Attempts made: Initial implementation with constraint formulation
Suspected issues: Constraint values differ by ~0.0029 from pycutest
Resources needed: Check constraint formulation, particularly equality constraints

This is an optimal control problem to minimize the energy spent to move a train
from the beginning of a track to its end in a given time. The train is slowed
down by drag (assumed to be quadratic in the velocity). The track follows the
slope of a hill with the geometry given by:

g(x) = -1/2(a_1 + a_ns) + 1/pi * SUM_{i=1}^{ns-1} (s_{i+1} - s_i) *
       arctan((x - z_i)/eps)

where z_i are breakpoints between track sections, s_i are slopes on sections,
and eps is a regularization parameter.

Variables (4*(N+1) total):
- X(i): position at time i (i=0 to N)
- V(i): velocity at time i (i=0 to N)
- UA(i): acceleration force at time i (i=0 to N)
- UB(i): braking force at time i (i=0 to N)

Source: adapted from
J. Kautsky and N. K. Nichols,
"OTEP-2: Optimal Train Energy Programme, mark 2",
Numerical Analysis Report NA/4/83,
Department of Mathematics, University of Reading, 1983.

SIF input: N. Nichols and Ph. Toint, April 1993

classification QOR2-MN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class TRAINH(AbstractConstrainedMinimisation):
    """TRAINH: Optimal train energy on hilly track."""

    # Problem parameters (default from SIF file)
    N: int = 1001  # Number of discretization points
    TIME: float = 4.8  # Travel time
    LENGTH: float = 6.0  # Track length

    # Physical parameters
    UAMAX: float = 10.0  # Maximum acceleration
    UBMIN: float = -2.0  # Maximum braking
    VMAX: float = 10.0  # Maximum train speed
    A: float = 0.3  # Drag constant term
    B: float = 0.14  # Drag linear term
    C: float = 0.16  # Drag quadratic term

    # Track geometry parameters
    NS: int = 3  # Number of track sections
    Z1: float = 2.0  # Breakpoint 1
    Z2: float = 4.0  # Breakpoint 2
    S1: float = 2.0  # Slope on section 1
    S2: float = 0.0  # Slope on section 2
    S3: float = -2.0  # Slope on section 3
    EPS: float = 0.05  # Regularization parameter
    PI: float = 3.1415926535

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def name(self) -> str:
        return "TRAINH"

    @property
    def n(self) -> int:
        return 4 * (self.N + 1)  # X, V, UA, UB for each time point

    @property
    def m(self) -> int:
        return 2 * self.N  # XEQ and VEQ constraints

    @property
    def args(self):
        return None

    def _compute_mesh(self):
        """Compute mesh-related constants."""
        h = self.TIME / self.N
        h_2 = h / 2.0
        return h, h_2

    def _compute_track_constants(self):
        """Compute constants related to track geometry."""
        h, h_2 = self._compute_mesh()

        # Sum of first and last slopes
        sums = self.S1 + self.S3
        avs = -0.5 * sums
        avsh = avs * h
        ah = self.A * h
        cnst = -ah - avsh  # Combined constant for VEQ

        h_2pi = h_2 / self.PI

        return cnst, h_2pi

    @property
    def y0(self):
        """Initial point from SIF file - fully vectorized."""
        n_points = self.N + 1

        # All variables in one go
        y0_vals = jnp.zeros(4 * n_points)

        # X values: linear from 0 to LENGTH
        y0_vals = y0_vals.at[:n_points].set(jnp.linspace(0.0, self.LENGTH, n_points))

        # V values: LENGTH/N for interior points, 0 at boundaries
        v_vals = jnp.full(n_points, self.LENGTH / self.N)
        v_vals = v_vals.at[jnp.array([0, n_points - 1])].set(0.0)
        y0_vals = y0_vals.at[n_points : 2 * n_points].set(v_vals)

        # UA values: UAMAX at start only
        y0_vals = y0_vals.at[2 * n_points].set(self.UAMAX)

        # UB values: UBMIN at end only
        y0_vals = y0_vals.at[4 * n_points - 1].set(self.UBMIN)

        return y0_vals

    @property
    def xl(self):
        """Lower bounds - fully vectorized."""
        n_points = self.N + 1

        # Start with -inf for all
        xl = jnp.full(4 * n_points, -jnp.inf)

        # Set specific bounds using index arrays
        # X bounds
        xl = xl.at[0].set(0.0)  # X(0) = 0
        xl = xl.at[n_points - 1].set(self.LENGTH)  # X(N) = LENGTH

        # V bounds
        xl = xl.at[n_points].set(0.0)  # V(0) = 0
        xl = xl.at[2 * n_points - 1].set(0.0)  # V(N) = 0

        # UA bounds
        xl = xl.at[2 * n_points].set(self.UAMAX)  # UA(0) = UAMAX
        xl = xl.at[2 * n_points + 1 : 3 * n_points - 1].set(0.0)  # UA interior >= 0
        xl = xl.at[3 * n_points - 1].set(0.0)  # UA(N) = 0

        # UB bounds
        xl = xl.at[3 * n_points].set(0.0)  # UB(0) = 0
        xl = xl.at[3 * n_points + 1 : 4 * n_points - 1].set(
            self.UBMIN
        )  # UB interior >= UBMIN
        xl = xl.at[4 * n_points - 1].set(self.UBMIN)  # UB(N) = UBMIN

        return xl

    @property
    def xu(self):
        """Upper bounds - fully vectorized."""
        n_points = self.N + 1

        # Start with +inf for all
        xu = jnp.full(4 * n_points, jnp.inf)

        # Set specific bounds using index arrays
        # X bounds
        xu = xu.at[0].set(0.0)  # X(0) = 0
        xu = xu.at[n_points - 1].set(self.LENGTH)  # X(N) = LENGTH

        # V bounds
        xu = xu.at[n_points].set(0.0)  # V(0) = 0
        xu = xu.at[2 * n_points - 1].set(0.0)  # V(N) = 0

        # UA bounds
        xu = xu.at[2 * n_points].set(self.UAMAX)  # UA(0) = UAMAX
        xu = xu.at[2 * n_points + 1 : 3 * n_points - 1].set(
            self.UAMAX
        )  # UA interior <= UAMAX
        xu = xu.at[3 * n_points - 1].set(0.0)  # UA(N) = 0

        # UB bounds
        xu = xu.at[3 * n_points].set(0.0)  # UB(0) = 0
        xu = xu.at[3 * n_points + 1 : 4 * n_points - 1].set(0.0)  # UB interior <= 0
        xu = xu.at[4 * n_points - 1].set(self.UBMIN)  # UB(N) = UBMIN

        return xu

    def objective(self, y, args):
        """Minimize energy: sum of UA(i) * V(i) * h for i=1 to N-1."""
        n_points = self.N + 1
        h, _ = self._compute_mesh()

        # Extract V and UA directly using slicing
        V = y[n_points : 2 * n_points]
        UA = y[2 * n_points : 3 * n_points]

        # Energy = sum of UA(i) * V(i) * h for i=1 to N-1
        # Use dot product for efficiency
        return h * jnp.dot(UA[1 : self.N], V[1 : self.N])

    def _compute_atan_terms(self, X):
        """Compute arctangent terms for track geometry."""
        # Breakpoints for track sections
        Z = jnp.array([self.Z1, self.Z2])  # Z1 and Z2

        # Slopes: differences between adjacent section slopes
        DS = jnp.array([self.S2 - self.S1, self.S3 - self.S2])  # S2-S1, S3-S2

        # Compute arctan terms for all positions and all breakpoints
        # Shape: (N+1, 2) for 2 breakpoints
        # atan_terms[i, j] = arctan((X[i] - Z[j]) / EPS)
        X_expanded = X[:, jnp.newaxis]  # Shape: (N+1, 1)
        Z_expanded = Z[jnp.newaxis, :]  # Shape: (1, 2)
        atan_vals = jnp.arctan((X_expanded - Z_expanded) / self.EPS)

        # Weight by slope differences and sum over breakpoints
        # DS shape: (2,), atan_vals shape: (N+1, 2)
        # Broadcast DS to match: DS[newaxis, :] * atan_vals
        weighted_sum = jnp.sum(DS[jnp.newaxis, :] * atan_vals, axis=1)

        return weighted_sum

    def constraint(self, y):
        """Constraints: position and velocity dynamics with hill - interleaved."""
        n_points = self.N + 1
        h, h_2 = self._compute_mesh()

        # Extract variables using direct slicing
        X = y[:n_points]
        V = y[n_points : 2 * n_points]
        UA = y[2 * n_points : 3 * n_points]
        UB = y[3 * n_points : 4 * n_points]

        # Pre-compute constants
        bh_2 = self.B * h_2
        one_plus_bh_2 = 1.0 + bh_2
        bh_2_minus_1 = bh_2 - 1.0
        ch_2 = self.C * h_2

        # Track geometry constants
        cnst, h_2pi = self._compute_track_constants()

        # Pre-compute sums for efficiency
        V_sum = V[1:] + V[:-1]
        UA_sum = UA[1:] + UA[:-1]
        UB_sum = UB[1:] + UB[:-1]
        V_sq_sum = V[:-1] ** 2 + V[1:] ** 2

        # XEQ constraints: X(i+1) - X(i) - h/2 * (V(i+1) + V(i)) = 0
        xeq = X[1:] - X[:-1] - h_2 * V_sum

        # Compute arctangent terms for hill geometry
        atan_terms = self._compute_atan_terms(X)

        # Sum arctangent terms for adjacent positions
        atan_sum = atan_terms[:-1] + atan_terms[1:]

        # VEQ constraints: velocity dynamics with hill geometry
        veq = (
            one_plus_bh_2 * V[1:]
            + bh_2_minus_1 * V[:-1]
            - h_2 * (UA_sum + UB_sum)
            + ch_2 * V_sq_sum
            + h_2pi * atan_sum  # Hill geometry contribution
            + cnst  # Combined constant term
        )

        # Interleave constraints as in pycutest: XEQ(0), VEQ(0), XEQ(1), VEQ(1), ...
        constraints = jnp.zeros(2 * self.N)
        constraints = constraints.at[::2].set(xeq)  # Even indices: XEQ
        constraints = constraints.at[1::2].set(veq)  # Odd indices: VEQ

        # Return tuple (equality_constraints, inequality_constraints)
        return constraints, None

    @property
    def bounds(self):
        """Return bounds as (xl, xu) tuple."""
        return self.xl, self.xu

    @property
    def expected_result(self):
        """Solution not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """From SIF file comment for N=1001: 12.307801327."""
        return jnp.array(12.307801327)
