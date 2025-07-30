import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Attempts made: Multiple attempts to fix data generation and constraint formulation
# Suspected issues: Complex differences in objective/constraint formulation vs pycutest
# Resources needed: Deep understanding of how pycutest handles orthogonal regression
class ORTHREGA(AbstractConstrainedMinimisation):
    """
    An orthogonal regression problem.

    The problem is to fit (orthogonally) an ellipse to a set of points
    in the plane.

    Source:
    M. Gulliksson,
    "Algorithms for nonlinear Least-squares with Applications to
    Orthogonal Regression",
    UMINF-178.90, University of Umea, Sweden, 1990.

    SIF input: Ph. Toint, June 1990.

    classification QQR2-AN-V-V

    Number of levels in the generation of the data points
    ( number of data points =     4**LEVELS
      number of variables   = 2 * 4**LEVELS + 5
      number of constraints =     4**LEVELS         )
    Default LEVELS = 3 (original value)
    """

    levels: int = 6  # Number of levels in data generation (default from SIF)
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, levels: int = 6):
        self.levels = levels

    @property
    def n(self) -> int:
        """Total number of variables: 2 * 4**levels + 5."""
        npts = 4**self.levels
        return 2 * npts + 5

    @property
    def m(self) -> int:
        """Number of constraints: 4**levels equality constraints."""
        return 4**self.levels

    def _generate_data_points(self):
        """Generate the data points for the ellipse fitting."""
        # Parameters for data generation
        a = 9.0
        b = 6.0
        cx = 0.5
        cy = 0.5
        pi = jnp.pi

        # Initialize with center point
        xd = [cx]
        yd = [cy]

        # Generate points level by level
        for i in range(1, self.levels + 1):
            # Store current points
            xz = xd.copy()
            yz = yd.copy()

            # Clear for new points
            xd = []
            yd = []

            # Generate 4 new points for each existing point
            for j in range(len(xz)):
                # Add 4 points around (xz[j], yz[j])
                xd.append(xz[j] + a)
                yd.append(yz[j] + a)

                xd.append(xz[j] + b)
                yd.append(yz[j] - b)

                xd.append(xz[j] - a)
                yd.append(yz[j] - a)

                xd.append(xz[j] - b)
                yd.append(yz[j] + b)

            # Update a and b for next level (divide by pi)
            a = a / pi
            b = b / pi

        return jnp.array(xd, dtype=jnp.float64), jnp.array(yd, dtype=jnp.float64)

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        npts = 4**self.levels
        y = jnp.zeros(self.n, dtype=jnp.float64)

        # Ellipse parameters
        y = y.at[0].set(1.0)  # h11
        y = y.at[1].set(0.0)  # h12
        y = y.at[2].set(1.0)  # h22
        y = y.at[3].set(0.0)  # g1
        y = y.at[4].set(0.0)  # g2

        # X and Y projections initialized to data points
        xd, yd = self._generate_data_points()
        y = y.at[5 : 5 + npts].set(xd)
        y = y.at[5 + npts :].set(yd)

        return y

    def objective(self, y: Array, args) -> Array:
        """Compute the objective function."""
        npts = 4**self.levels

        # Extract variables
        x = y[5 : 5 + npts]
        y_points = y[5 + npts :]

        # Get data points
        xd, yd = self._generate_data_points()

        # Objective: sum of squared distances to data points
        obj = jnp.sum((x - xd) ** 2 + (y_points - yd) ** 2)

        return obj

    def constraint(self, y: Array):
        """Compute the equality and inequality constraints."""
        npts = 4**self.levels

        # Extract variables
        h11 = y[0]
        h12 = y[1]
        h22 = y[2]
        g1 = y[3]
        g2 = y[4]
        x = y[5 : 5 + npts]
        y_points = y[5 + npts :]

        # Equality constraints
        # E(i): ellipse constraint equation
        eq_constraints = (
            h11 * x**2
            + 2.0 * h12 * x * y_points
            + h22 * y_points**2
            - 2.0 * g1 * x
            - 2.0 * g2 * y_points
            + 1.0
        )

        # No inequality constraints
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
        # From SIF file comments:
        # SOLTN(3) = 350.29936756
        # SOLTN(4) = 1414.0524915
        return jnp.array(350.29936756)
