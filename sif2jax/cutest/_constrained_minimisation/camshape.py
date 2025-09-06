import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made: 4 attempts at constraint formulation fixes
# Suspected issues:
#   - Constraint value sign mismatches with pycutest (opposite signs)
#   - Complex range constraint interpretation from SIF XG constraints
#   - Edge constraint formulation may be incorrect
# Resources needed: SIF XG/RANGES constraint expertise, pycutest comparison


class CAMSHAPE(AbstractConstrainedMinimisation):
    """Cam shape optimization problem for maximizing valve opening area.

    Maximize the area of the valve opening for one rotation of a convex cam
    with constraints on the curvature and on the radius of the cam.

    This is problem 4 in the COPS (Version 2) collection of
    E. Dolan and J. More', "Benchmarking Optimization Software with COPS",
    Argonne National Labs Technical Report ANL/MCS-246 (2000).

    SIF input: Nick Gould, November 2000.

    Classification: LOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters from SIF file
    n_points: int = 800  # Number of discretization points
    rv: float = 1.0  # Design parameter related to valve shape
    rmax: float = 2.0  # Maximum allowed radius
    rmin: float = 1.0  # Minimum allowed radius
    alpha: float = 1.5  # Curvature limit parameter

    # Computed parameters as properties to avoid mutable defaults
    @property
    def rav(self):
        return (self.rmin + self.rmax) * 0.5  # Average radius

    @property
    def pi(self):
        return 4.0 * jnp.arctan(1.0)  # Approximation of pi

    @property
    def dtheta(self):
        return (
            self.pi * 2.0 / (5 * (self.n_points + 1))
        )  # Angle between discretization points

    @property
    def pirv_n(self):
        return self.pi * self.rv / self.n_points  # Objective coefficient

    @property
    def cdtheta(self):
        return jnp.cos(self.dtheta)

    @property
    def cdtheta2(self):
        return 2.0 * self.cdtheta

    @property
    def adtheta(self):
        return self.alpha * self.dtheta

    @property
    def rmin2cd(self):
        return self.rmin * self.cdtheta2

    @property
    def rmax2cd(self):
        return self.rmax * self.cdtheta2

    @property
    def rmin2(self):
        return self.rmin * self.rmin

    @property
    def n(self):
        """Number of variables: N discretization points."""
        return self.n_points

    def objective(self, y, args):
        """Minimize negative valve area: -((PI*RV)/N)*sum(R[i])."""
        del args
        # Original objective: maximize ((PI*RV)/N)*sum(R[i])
        # Converted to minimize: -((PI*RV)/N)*sum(R[i])
        return -self.pirv_n * jnp.sum(y)

    def constraint(self, y):
        """Convexity and curvature constraints."""
        r = y  # Radius values R(1)...R(N)
        n = self.n_points

        inequality_constraints = []

        # Convexity constraints for i in 2..N-1:
        # -R[i-1]*R[i] - R[i]*R[i+1] + R[i-1]*R[i+1]*2CDTHETA <= 0
        for i in range(1, n - 1):  # i = 1 to n-2 (0-indexed)
            constraint = (
                -r[i - 1] * r[i] - r[i] * r[i + 1] + r[i - 1] * r[i + 1] * self.cdtheta2
            )
            inequality_constraints.append(constraint)

        # Convex edge constraints:
        # E1: -RMIN*R[1] - R[1]*R[2] + RMIN*R[2]*2CDTHETA <= 0
        constraint_e1 = -self.rmin * r[0] - r[0] * r[1] + self.rmin2cd * r[1]
        inequality_constraints.append(constraint_e1)

        # E2: -RMIN^2 - RMIN*R[1] + RMIN*R[1]*2CDTHETA <= 0
        constraint_e2 = -self.rmin2 - self.rmin * r[0] + self.rmin2cd * r[0]
        inequality_constraints.append(constraint_e2)

        # E3: -R[N-1]*R[N] - R[N]*RMAX + R[N-1]*RMAX*2CDTHETA <= 0
        constraint_e3 = (
            -r[n - 2] * r[n - 1] - r[n - 1] * self.rmax + r[n - 2] * self.rmax2cd
        )
        inequality_constraints.append(constraint_e3)

        # E4: -2*RMAX*R[N] + R[N]^2*2CDTHETA <= 0
        constraint_e4 = -2 * self.rmax * r[n - 1] + r[n - 1] ** 2 * self.cdtheta2
        inequality_constraints.append(constraint_e4)

        # Curvature constraints (ranges -ALPHA*DTHETA <= constraint <= ALPHA*DTHETA):
        # From SIF: these are XG (general constraints) with RANGES, which means they're
        # handled as single constraints, not as two inequalities each

        # CU(0): R[1] - RMIN (with range)
        cu0 = r[0] - self.rmin + self.adtheta  # Shift to make constraint >= 0
        inequality_constraints.append(-cu0)  # <= 0 format

        # CU(i) for i in 1..N-1: R[i+1] - R[i] (with range)
        for i in range(n - 1):  # i = 0 to n-2 (0-indexed)
            cui = r[i + 1] - r[i] + self.adtheta  # Shift to make constraint >= 0
            inequality_constraints.append(-cui)  # <= 0 format

        # CU(N): RMAX - R[N] (with range)
        cun = self.rmax - r[n - 1] + self.adtheta  # Shift to make constraint >= 0
        inequality_constraints.append(-cun)  # <= 0 format

        inequality_constraints = jnp.array(inequality_constraints)

        return None, inequality_constraints  # No equality constraints

    @property
    def y0(self):
        """Initial guess: average radius for all discretization points."""
        return jnp.full(self.n_points, self.rav)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Variable bounds: R(i) ∈ [RMIN, RMAX]."""
        lower = jnp.full(self.n_points, self.rmin)
        upper = jnp.full(self.n_points, self.rmax)
        return lower, upper

    @property
    def expected_result(self):
        """Expected solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF: -4.2743D+00 for NH=800
        return jnp.array(-4.2743)
