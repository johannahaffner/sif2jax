import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


class ORTHREGB(AbstractConstrainedMinimisation):
    """
    An orthogonal regression problem.

    The problem is to fit (orthogonally) an ellipse to a set of 6 points
    in the 3D space. These points are compatible with this constraint.

    Source:
    M. Gulliksson,
    "Algorithms for nonlinear Least-squares with Applications to
    Orthogonal Regression",
    UMINF-178.90, University of Umea, Sweden, 1990.

    SIF input: Ph. Toint, June 1990.
               correction by Ph. Shott, Jan 1995.

    classification QQR2-AN-27-6

    Number of variables: 27 (9 ellipse parameters + 18 point projections)
    Number of constraints: 6 (ellipse constraints)
    """

    npts: int = 6  # Number of data points (fixed)
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self) -> int:
        """Total number of variables: 9 ellipse parameters + 3 * NPTS projections."""
        return 9 + 3 * self.npts

    @property
    def m(self) -> int:
        """Number of constraints: NPTS equality constraints."""
        return self.npts

    def _generate_data_points(self):
        """Generate the 6 data points."""
        # Parameters from SIF
        a, b, c = 9.0, 6.0, 7.0
        cx, cy, cz = 0.5, 0.5, 0.5

        # The 6 data points as defined in the SIF file
        xd = jnp.array(
            [
                cx + a,  # Point 1: (cx+a, cy+a, cz)
                cx + b,  # Point 2: (cx+b, cy-b, cz)
                cx - a,  # Point 3: (cx-a, cy-a, cz)
                cx - b,  # Point 4: (cx-b, cy+b, cz)
                cx,  # Point 5: (cx, cy, cz+c)
                cx,  # Point 6: (cx, cy, cz-c)
            ],
            dtype=jnp.float64,
        )

        yd = jnp.array(
            [
                cy + a,  # Point 1
                cy - b,  # Point 2
                cy - a,  # Point 3
                cy + b,  # Point 4
                cy,  # Point 5
                cy,  # Point 6
            ],
            dtype=jnp.float64,
        )

        zd = jnp.array(
            [
                cz,  # Point 1
                cz,  # Point 2
                cz,  # Point 3
                cz,  # Point 4
                cz + c,  # Point 5
                cz - c,  # Point 6
            ],
            dtype=jnp.float64,
        )

        return xd, yd, zd

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        y = jnp.zeros(self.n, dtype=jnp.float64)

        # Ellipse parameters (H11, H12, H13, H22, H23, H33, G1, G2, G3)
        y = y.at[0].set(1.0)  # H11
        y = y.at[1].set(0.0)  # H12
        y = y.at[2].set(0.0)  # H13
        y = y.at[3].set(1.0)  # H22
        y = y.at[4].set(0.0)  # H23
        y = y.at[5].set(1.0)  # H33
        y = y.at[6].set(0.0)  # G1
        y = y.at[7].set(0.0)  # G2
        y = y.at[8].set(0.0)  # G3

        # Point projections initialized to data points
        xd, yd, zd = self._generate_data_points()
        for i in range(self.npts):
            y = y.at[9 + 3 * i].set(xd[i])  # X(i)
            y = y.at[9 + 3 * i + 1].set(yd[i])  # Y(i)
            y = y.at[9 + 3 * i + 2].set(zd[i])  # Z(i)

        return y

    def objective(self, y: Array, args) -> Array:
        """Compute the objective function."""
        # Get data points
        xd, yd, zd = self._generate_data_points()

        # Extract projected points
        obj = jnp.array(0.0)
        for i in range(self.npts):
            x_proj = y[9 + 3 * i]
            y_proj = y[9 + 3 * i + 1]
            z_proj = y[9 + 3 * i + 2]

            # Sum of squared distances to data points
            obj += (x_proj - xd[i]) ** 2 + (y_proj - yd[i]) ** 2 + (z_proj - zd[i]) ** 2

        return obj

    def constraint(self, y: Array):
        """Compute the equality constraints."""
        # Extract ellipse parameters
        h11, h12, h13 = y[0], y[1], y[2]
        h22, h23, h33 = y[3], y[4], y[5]
        g1, g2, g3 = y[6], y[7], y[8]

        eq_constraints = []

        for i in range(self.npts):
            x = y[9 + 3 * i]
            y_coord = y[9 + 3 * i + 1]
            z = y[9 + 3 * i + 2]

            # Ellipse constraint: H11*x^2 + 2*H12*x*y + H22*y^2 + 2*H13*x*z +
            # 2*H23*y*z + H33*z^2 - 2*G1*x - 2*G2*y - 2*G3*z + 1 = 0
            constraint_val = (
                h11 * x**2
                + 2.0 * h12 * x * y_coord
                + h22 * y_coord**2
                + 2.0 * h13 * x * z
                + 2.0 * h23 * y_coord * z
                + h33 * z**2
                - 2.0 * g1 * x
                - 2.0 * g2 * y_coord
                - 2.0 * g3 * z
                + 1.0
            )
            eq_constraints.append(constraint_val)

        eq_constraints = jnp.array(eq_constraints)
        ineq_constraints = None

        return eq_constraints, ineq_constraints

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """All variables are free."""
        return None

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the objective and constraint functions."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Not explicitly given in the SIF file
        return jnp.zeros(self.n, dtype=jnp.float64)

    @property
    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # Should be near zero if fitting is perfect
        return jnp.array(0.0)
