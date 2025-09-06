import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made: 6 attempts at dimension and starting point fixes
# Suspected issues:
#   - Starting point mismatch with pycutest (variables layout unclear)
#   - Constraint evaluation errors (first constraint has 0.36 error)
#   - Gradient evaluation incorrect at starting point
# Resources needed: SIF file interpretation expertise, pycutest debugging


class CATENARY(AbstractConstrainedMinimisation):
    """An erroneous but interesting version of the classical hanging catenary problem.

    The catenary consists of N+1 beams of length BL, with the first beam fixed
    at the origin and the final beam fixed at a fraction FRACT of the total
    length of all beams.

    The correct version is given by problem CATENA.

    Source: K. Veselic, "De forma catenarum in campo gravitatis pendentium",
    Klasicna Gimnazija u Zagreb, Zagreb, 1987.

    SIF input: Ph. L. Toint, May 1993.

    Classification: LQR2-AY-V-V

    This problem is non-convex.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters from SIF file
    n_plus_1: int = 1000  # Number of beams = N+1
    n_beams: int = n_plus_1 - 1  # N
    gamma: float = 9.81  # gravity
    tmass: float = 500.0  # total mass of the N+1 beams
    bl: float = 1.0  # beam length
    fract: float = 0.6  # shortening factor

    # Computed parameters
    length: float = bl * n_plus_1 * fract
    mass: float = tmass / n_plus_1
    mg: float = mass * gamma
    mg_half: float = mg * 0.5
    blsq: float = bl * bl

    @property
    def n(self):
        """Number of variables: 3*(N+2)"""
        return 3 * (self.n_plus_1 + 1)  # 3*(N+1+1) = 3*(N+2)

    def objective(self, y, args):
        """Minimize weighted sum of Y coordinates (gravitational potential energy)."""
        del args

        # Variables are X(0),...,X(N+1), Y(0),...,Y(N+1), Z(0),...,Z(N+1)
        n_points = self.n_plus_1 + 1  # N+2 points total
        y_coords = y[n_points : 2 * n_points]  # Y coordinates

        # Objective: MG/2 * Y(0) + MG * sum(Y(1)...Y(N)) + MG/2 * Y(N+1)
        obj = self.mg_half * y_coords[0]  # Y(0)
        obj += self.mg * jnp.sum(y_coords[1:-1])  # Y(1) to Y(N)
        obj += self.mg_half * y_coords[-1]  # Y(N+1)

        return obj

    def constraint(self, y):
        """Beam length constraints."""
        n_points = self.n_plus_1 + 1  # N+2 points total

        # Extract coordinates
        x_coords = y[:n_points]  # X(0) to X(N+1)
        y_coords = y[n_points : 2 * n_points]  # Y(0) to Y(N+1)
        z_coords = y[2 * n_points : 3 * n_points]  # Z(0) to Z(N+1)

        # Beam length constraints: ||joint[i] - joint[i-1]||^2 = BL^2
        # For i = 1 to N+1 (total of N+1 constraints)
        equality_constraints = []

        for i in range(1, n_points):
            dx = x_coords[i] - x_coords[i - 1]
            dy = y_coords[i] - y_coords[i - 1]
            dz = z_coords[i] - z_coords[i - 1]

            # Constraint: dx^2 + dy^2 + dz^2 = BLSQ
            # In standard form: dx^2 + dy^2 + dz^2 - BLSQ = 0
            constraint = dx**2 + dy**2 + dz**2 - self.blsq
            equality_constraints.append(constraint)

        equality_constraints = jnp.array(equality_constraints)

        return equality_constraints, None  # No inequality constraints

    @property
    def y0(self):
        """Initial guess with linear spacing for X coordinates."""
        n_points = self.n_plus_1 + 1  # N+2 points total

        # Initialize coordinates
        x_coords = jnp.zeros(n_points)
        y_coords = jnp.zeros(n_points)
        z_coords = jnp.zeros(n_points)

        # From SIF: TMP = LENGTH / RN+1, VAL = TMP * I, then X(I) = VAL for I=1 to N+1
        # Note: CATENARY.SIF says X(I) = TMP, but based on pycutest this should be VAL
        tmp = self.length / self.n_plus_1
        for i in range(1, n_points):
            val = tmp * i
            x_coords = x_coords.at[i].set(val)

        # Concatenate all coordinates
        return jnp.concatenate([x_coords, y_coords, z_coords])

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def bounds(self):
        """Variable bounds."""
        n_points = self.n_plus_1 + 1  # N+2 points total
        n_vars = 3 * n_points

        # Default: all variables unbounded except fixed points
        lower = jnp.full(n_vars, -jnp.inf)
        upper = jnp.full(n_vars, jnp.inf)

        # Fix initial joint at origin: X(0)=0, Y(0)=0, Z(0)=0
        lower = lower.at[0].set(0.0)  # X(0)
        upper = upper.at[0].set(0.0)
        lower = lower.at[n_points].set(0.0)  # Y(0)
        upper = upper.at[n_points].set(0.0)
        lower = lower.at[2 * n_points].set(0.0)  # Z(0)
        upper = upper.at[2 * n_points].set(0.0)

        # Fix final joint X position: X(N+1) = LENGTH
        final_x_idx = n_points - 1
        lower = lower.at[final_x_idx].set(self.length)
        upper = upper.at[final_x_idx].set(self.length)

        return lower, upper

    @property
    def expected_result(self):
        """Expected solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None
