import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class PENTAGON(AbstractConstrainedMinimisation):
    """
    PENTAGON problem.

    An approximation to the problem of finding 3 points in a 2D
    pentagon whose minimal distance is maximal.

    Source:
    M.J.D. Powell,
    " TOLMIN: a Fortran package for linearly constrained
    optimization problems",
    Report DAMTP 1989/NA2, University of Cambridge, UK, 1989.

    SIF input: Ph. Toint, May 1990.

    classification OLR2-AY-6-15
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, y1, x2, y2, x3, y3 = y

        # Distances squared
        d12_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
        d13_sq = (x1 - x3) ** 2 + (y1 - y3) ** 2
        d23_sq = (x2 - x3) ** 2 + (y2 - y3) ** 2

        # Objective: sum of 1/(d^2)^8 for each distance
        # From SIF: F = 1.0 / D**8 where D = DX*DX + DY*DY
        return 1.0 / d12_sq**8 + 1.0 / d13_sq**8 + 1.0 / d23_sq**8

    def y0(self):
        # Starting point
        return jnp.array([-1.0, 0.0, 0.0, -1.0, 1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in detail in SIF file
        return None

    def expected_objective_value(self):
        # Solution value provided as 1.36521631D-04
        return jnp.array(1.36521631e-04)

    def num_variables(self):
        return 6

    def bounds(self):
        # All variables are free (unbounded)
        return jnp.array([-jnp.inf] * 6), jnp.array([jnp.inf] * 6)

    def constraint(self, y):
        x1, y1, x2, y2, x3, y3 = y

        # Pentagon vertices
        # Pentagon with vertices at (cos(2*pi*j/5), sin(2*pi*j/5)) for j = 0, 1, 2, 3, 4
        two_pi_over_5 = 1.2566371
        cos_vals = jnp.array([jnp.cos(j * two_pi_over_5) for j in range(5)])
        sin_vals = jnp.array([jnp.sin(j * two_pi_over_5) for j in range(5)])

        # Inequality constraints: each point must be inside the pentagon
        # For each point i and each edge j: C(i,j) <= 1
        # C(i,j) = x(i) * cos(j * 2*pi/5) + y(i) * sin(j * 2*pi/5)
        # pycutest returns raw values: C(i,j) - 1.0
        ineq_constraint = jnp.array(
            [
                # Point 1 constraints
                (x1 * cos_vals[0] + y1 * sin_vals[0]) - 1.0,
                (x1 * cos_vals[1] + y1 * sin_vals[1]) - 1.0,
                (x1 * cos_vals[2] + y1 * sin_vals[2]) - 1.0,
                (x1 * cos_vals[3] + y1 * sin_vals[3]) - 1.0,
                (x1 * cos_vals[4] + y1 * sin_vals[4]) - 1.0,
                # Point 2 constraints
                (x2 * cos_vals[0] + y2 * sin_vals[0]) - 1.0,
                (x2 * cos_vals[1] + y2 * sin_vals[1]) - 1.0,
                (x2 * cos_vals[2] + y2 * sin_vals[2]) - 1.0,
                (x2 * cos_vals[3] + y2 * sin_vals[3]) - 1.0,
                (x2 * cos_vals[4] + y2 * sin_vals[4]) - 1.0,
                # Point 3 constraints
                (x3 * cos_vals[0] + y3 * sin_vals[0]) - 1.0,
                (x3 * cos_vals[1] + y3 * sin_vals[1]) - 1.0,
                (x3 * cos_vals[2] + y3 * sin_vals[2]) - 1.0,
                (x3 * cos_vals[3] + y3 * sin_vals[3]) - 1.0,
                (x3 * cos_vals[4] + y3 * sin_vals[4]) - 1.0,
            ]
        )

        # Equality constraints: none
        eq_constraint = None

        return eq_constraint, ineq_constraint
