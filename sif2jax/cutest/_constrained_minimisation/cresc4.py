import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made: Separated constraints, fixed constraint signs
# Suspected issues: Complex crescent area formula (SC element) with arccos
# Additional resources needed: Full SC element type or simplified formula
class CRESC4(AbstractConstrainedMinimisation):
    """CRESC4 problem - finding crescent of smallest area containing 4 points.

    This problem consists in finding the crescent of smallest area containing
    a set of points given in the plane. A crescent is defined as the area
    contained in circle 2 but not in circle 1.

    The parametrization used:
    - (C2x, C2y) = (C1x, C1y) + a * d * (cos(t), sin(t))
    - r1 = a * d + r
    - r2 = (a + 1) * d + r

    with bounds: a >= 1, 0 <= t <= 2*pi, r >= 0, 0 <= d <= 1

    SIF input: Ph. Toint, June 1993.

    Classification: OOR2-MY-6-8
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 6

    @property
    def m(self):
        """Number of constraints."""
        return 8  # 2 constraints per point, 4 points

    def objective(self, y, args):
        """Compute the objective (area of crescent)."""
        del args

        # Extract variables
        v1, w1, d, a, t, r = y

        # The crescent area calculation is complex and involves
        # the intersection of two circles. The SIF file uses a
        # sophisticated formula with arccos functions.
        # For now, let's use the simple difference of areas:

        # Radii
        r1 = a * d + r
        r2 = (a + 1.0) * d + r

        # Simple area calculation: Area = pi * (r2^2 - r1^2)
        # But actually, the true area of a crescent formed by two
        # intersecting circles requires accounting for the overlap.
        # The SIF file computes this using element type SC.

        # For now, return a placeholder that gives reasonable values
        # TODO: Implement the full SC element computation
        obj = jnp.pi * (r2**2 - r1**2)

        return obj

    def constraint(self, y):
        """Compute the constraints separated into equalities and inequalities."""
        # Extract variables
        v1, w1, d, a, t, r = y

        # Data points
        x_data = jnp.array([1.0, 0.0, 0.0, 0.5])
        y_data = jnp.array([0.0, 1.0, -1.0, 0.0])

        # Center of circle 2
        v2 = v1 + a * d * jnp.cos(t)
        w2 = w1 + a * d * jnp.sin(t)

        # Radii
        r1 = a * d + r
        r2 = (a + 1.0) * d + r

        # Constraints: for each point i
        # point must be inside circle 2: (xi - v2)^2 + (yi - w2)^2 <= r2^2
        # point must be outside circle 1: (xi - v1)^2 + (yi - w1)^2 >= r1^2

        constraints = []

        for i in range(4):
            xi = x_data[i]
            yi = y_data[i]

            # Inside circle 2 (inequality <= 0, convert to >= 0)
            dist2_sq = (xi - v2) ** 2 + (yi - w2) ** 2
            constraints.append(r2**2 - dist2_sq)

            # Outside circle 1 (inequality >= 0)
            dist1_sq = (xi - v1) ** 2 + (yi - w1) ** 2
            constraints.append(dist1_sq - r1**2)

        # All constraints are inequalities, no equalities
        return None, jnp.array(constraints)

    def y0(self):
        """Initial guess."""
        return jnp.array([-40.0, 5.0, 1.0, 2.0, 1.5, 0.75])

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds."""
        lower = jnp.array([-jnp.inf, -jnp.inf, 1e-8, 1.0, 0.0, 0.39])
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf, 6.2831852, jnp.inf])
        return lower, upper

    def expected_result(self):
        """Expected optimal solution."""
        return jnp.array([-0.75, 0.0, 0.5, 1.5, 0.0, 0.5])

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file comment
        return jnp.array(0.87189692)
