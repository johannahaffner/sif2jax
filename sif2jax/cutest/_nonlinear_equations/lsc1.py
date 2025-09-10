import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class LSC1(AbstractNonlinearEquations):
    """Fit a circle to a set of 2D points: case 1, data points surround the circle.

    Given as an inconsistent set of nonlinear equations.

    Source: Problem from the SciPy cookbook
    http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    SIF input: Nick Gould, Nov 2016

    Classification: NOR2-MN-3-6
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    # Data points as class fields
    x_data = jnp.array([9.0, 35.0, -13.0, 10.0, 23.0, 0.0])
    y_data = jnp.array([34.0, 10.0, 6.0, -14.0, 27.0, -10.0])

    @property
    def n_var(self):
        """Number of variables: X, Y, R."""
        return 3

    @property
    def n_con(self):
        """Number of constraints: 6 equations."""
        return 6

    def constraint(self, y):
        """Compute the nonlinear equations.

        For each data point (px, py), the equation is:
        sqrt((X - px)^2 + (Y - py)^2) - R = 0
        """
        x_center, y_center, r = y[0], y[1], y[2]

        # Compute distances from center to each data point
        dx = x_center - self.x_data
        dy = y_center - self.y_data
        distances = jnp.sqrt(dx**2 + dy**2)

        # Equations: distance - R = 0
        residuals = distances - r

        # Return as equality constraints (None for inequality part)
        return residuals, None

    @property
    def y0s(self):
        """Starting points."""
        # START1
        start1 = jnp.array([105.0, 96.0, 230.0])
        # START2
        start2 = jnp.array([10.5, 9.6, 23.0])
        return {0: start1, 1: start2}

    @property
    def y0(self):
        """Default initial guess (START1)."""
        return self.y0s[0]

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """No explicit bounds (FR EXAMPLEX 'DEFAULT')."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return None
