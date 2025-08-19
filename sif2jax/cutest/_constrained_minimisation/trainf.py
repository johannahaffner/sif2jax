"""
TRAINF: Optimal Train Energy Problem (Flat Track).

This is an optimal control problem to minimize the energy spent to move a train
from the beginning of a flat track to its end in a given time. The train
is slowed down by drag (assumed to be quadratic in the velocity).
The control variables are the acceleration force (UA) and the braking
force (UB) applied on the train.

Variables (4*(N+1) total):
- X(i): position at time i (i=0 to N)
- V(i): velocity at time i (i=0 to N)
- UA(i): acceleration force at time i (i=0 to N)
- UB(i): braking force at time i (i=0 to N)

Source:
J. Kautsky and N. K. Nichols,
"OTEP-2: Optimal Train Energy Programme, mark 2",
Numerical Analysis Report NA/4/83,
Department of Mathematics, University of Reading, 1983.

SIF input: N. Nichols and Ph. Toint, April 1993

classification QQR2-MN-V-V
"""

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class TRAINF(AbstractConstrainedMinimisation):
    """TRAINF: Optimal train energy on flat track."""

    # Problem parameters (default from SIF file)
    N: int = 1001  # Number of discretization points
    TIME: float = 1.5  # Travel time
    LENGTH: float = 2.0  # Track length

    # Physical parameters
    UAMAX: float = 10.0  # Maximum acceleration
    UBMIN: float = -2.0  # Maximum braking
    VMAX: float = 10.0  # Maximum train speed
    A: float = 0.3  # Drag constant term
    B: float = 0.14  # Drag linear term
    C: float = 0.16  # Drag quadratic term

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def name(self) -> str:
        return "TRAINF"

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

    def _extract_variables(self, y):
        """Extract X, V, UA, UB from flattened variable vector."""
        n_points = self.N + 1
        X = y[:n_points]
        V = y[n_points : 2 * n_points]
        UA = y[2 * n_points : 3 * n_points]
        UB = y[3 * n_points : 4 * n_points]
        return X, V, UA, UB

    @property
    def y0(self):
        """Initial point from SIF file."""
        n_points = self.N + 1

        # Initialize X values: linear from 0 to LENGTH
        X = jnp.linspace(0.0, self.LENGTH, n_points)

        # Initialize V values: LENGTH/N for interior points, 0 at boundaries
        V = jnp.ones(n_points) * (self.LENGTH / self.N)
        V = V.at[0].set(0.0)
        V = V.at[-1].set(0.0)

        # Initialize UA values: UAMAX at start, 0 elsewhere
        UA = jnp.zeros(n_points)
        UA = UA.at[0].set(self.UAMAX)

        # Initialize UB values: UBMIN at end, 0 elsewhere
        UB = jnp.zeros(n_points)
        UB = UB.at[-1].set(self.UBMIN)

        # Concatenate all variables
        return jnp.concatenate([X, V, UA, UB])

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

        # Extract V and UA directly using slicing - avoid unnecessary extraction
        V = y[n_points : 2 * n_points]
        UA = y[2 * n_points : 3 * n_points]

        # Energy = sum of UA(i) * V(i) * h for i=1 to N-1
        # Use dot product for efficiency
        return h * jnp.dot(UA[1 : self.N], V[1 : self.N])

    def constraint(self, y):
        """Constraints: position and velocity dynamics - interleaved as in pycutest."""
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
        ah = self.A * h
        ch_2 = self.C * h_2

        # Pre-compute sums for efficiency
        V_sum = V[1:] + V[:-1]
        UA_sum = UA[1:] + UA[:-1]
        UB_sum = UB[1:] + UB[:-1]
        V_sq_sum = V[:-1] ** 2 + V[1:] ** 2

        # XEQ constraints: X(i+1) - X(i) - h/2 * (V(i+1) + V(i)) = 0
        xeq = X[1:] - X[:-1] - h_2 * V_sum

        # VEQ constraints: velocity dynamics
        veq = (
            one_plus_bh_2 * V[1:]
            + bh_2_minus_1 * V[:-1]
            - h_2 * (UA_sum + UB_sum)
            + ch_2 * V_sq_sum
            + ah
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
        """From SIF file comment: 3.09751881012."""
        return jnp.array(3.09751881012)
