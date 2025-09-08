import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class LSC2(AbstractNonlinearEquations):
    """Fit a circle to a set of 2D points: case 2, data points in a small arc.

    Given as an inconsistent set of nonlinear equations.

    Source: Problem from the SciPy cookbook
    http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    SIF input: Nick Gould, Nov 2016

    Classification: NOR2-MN-3-6
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    # Data points as class fields (from LSC2.SIF)
    x_data = jnp.array([36.0, 36.0, 19.0, 18.0, 33.0, 26.0])
    y_data = jnp.array([14.0, 10.0, 28.0, 31.0, 18.0, 26.0])

    @property
    def n_var(self):
        """Number of variables: X, Y, R."""
        return jnp.array(3)

    @property
    def n_con(self):
        """Number of constraints: 6 equations."""
        return jnp.array(6)

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
        # START1 (from LSC2.SIF)
        start1 = jnp.array([98.0, 36.0, 270.0])
        # START2
        start2 = jnp.array([9.8, 3.6, 27.0])
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
