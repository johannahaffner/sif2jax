import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed - constraint values don't match pycutest
# The constraint formulation seems correct based on AMPL model, but
# pycutest returns very different values at the starting point
# The SIF file indicates constraints should be (x-cx)^2 + (y-cy)^2 >= r
# but pycutest values suggest a different formulation
class AIRPORT(AbstractConstrainedMinimisation):
    """Localization of airports in Brazil.

    This problem concerns the localization of airports in Brazil.
    The goal is to find optimal positions for 42 points (airports) on 42 balls
    (Brazilian cities), where each ball has center coordinates and a radius
    such that the balls are disjoint.

    The objective is to minimize the sum of squared Euclidean distances between
    all pairs of airports.

    Source: Problem provided by Antonio Carlos Moretti,
    State University of Campinas, Brazil

    SIF input: A.L. Tits and J.L. Zhou, Spring 1989.

    Classification: SQR2-MN-84-42
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n_airports: int = 42
    cx: Array = eqx.field(init=False)
    cy: Array = eqx.field(init=False)
    r: Array = eqx.field(init=False)

    def __init__(self):
        """Initialize the AIRPORT problem with city center coordinates and radii."""
        # Center x-coordinates of the 42 cities
        self.cx = jnp.array(
            [
                0.8,
                0.2,
                2.4,
                5.7,
                5.7,
                2.4,
                4.0,
                4.7,
                4.0,
                2.5,
                0.1,
                -1.2,
                -3.5,
                -4.0,
                -9.0,
                -2.5,
                -2.0,
                -2.0,
                -1.0,
                -0.5,
                2.0,
                1.0,
                1.8,
                0.0,
                -1.5,
                -0.5,
                -0.5,
                3.0,
                4.0,
                4.0,
                3.5,
                4.0,
                3.0,
                1.0,
                -1.0,
                -1.0,
                -2.7,
                -2.5,
                -0.5,
                1.5,
                2.0,
                3.5,
            ]
        )

        # Center y-coordinates of the 42 cities
        self.cy = jnp.array(
            [
                8.0,
                7.0,
                5.5,
                4.0,
                2.0,
                0.5,
                0.0,
                -1.5,
                -2.0,
                -2.0,
                -1.5,
                -1.5,
                -1.0,
                -0.5,
                1.0,
                0.5,
                1.0,
                1.5,
                2.5,
                3.5,
                4.5,
                3.0,
                2.0,
                1.0,
                1.8,
                2.5,
                3.5,
                4.0,
                3.0,
                1.0,
                0.0,
                -2.5,
                -3.0,
                -3.0,
                -3.5,
                -5.0,
                -4.0,
                -7.5,
                -6.0,
                -5.0,
                -3.0,
                -2.0,
            ]
        )

        # Radii of the 42 cities
        self.r = jnp.array(
            [
                0.09,
                0.45,
                0.51,
                0.67,
                0.67,
                0.49,
                0.02,
                0.45,
                0.40,
                0.17,
                0.20,
                0.67,
                0.34,
                0.40,
                0.67,
                0.24,
                0.20,
                0.20,
                0.60,
                0.40,
                0.67,
                0.67,
                0.05,
                0.67,
                0.52,
                0.53,
                0.46,
                0.67,
                0.60,
                0.67,
                0.40,
                0.31,
                0.20,
                0.67,
                0.50,
                0.62,
                0.26,
                0.15,
                0.60,
                0.40,
                0.40,
                0.40,
            ]
        )

    @property
    def n(self):
        """Total number of variables (2 * number of airports)."""
        return 2 * self.n_airports

    @property
    def m(self):
        """Total number of constraints."""
        return self.n_airports

    def objective(self, y, args):
        """Compute the sum of squared distances between all pairs of airports."""
        del args
        # Extract x and y coordinates
        x = y[: self.n_airports]
        y_coords = y[self.n_airports :]

        # Compute sum of squared distances between all pairs
        # This vectorized approach should produce the correct Hessian structure
        obj = jnp.array(0.0)
        for i in range(self.n_airports):
            for j in range(i + 1, self.n_airports):
                dx = x[i] - x[j]
                dy = y_coords[i] - y_coords[j]
                obj = obj + dx * dx + dy * dy

        return obj

    def constraint(self, y):
        """Compute the inequality constraints (airport must be within city ball)."""
        # Extract x and y coordinates
        x = y[: self.n_airports]
        y_coords = y[self.n_airports :]

        # Compute constraints: (x - cx)^2 + (y - cy)^2 >= r for each city
        # This is equivalent to dist_sq - r >= 0
        constraints = []
        for i in range(self.n_airports):
            dx = x[i] - self.cx[i]
            dy = y_coords[i] - self.cy[i]
            dist_sq = dx * dx + dy * dy
            # Constraint: dist_sq - r >= 0
            constraints.append(dist_sq - self.r[i])

        return None, jnp.array(constraints)

    def y0(self):
        """Initial guess for variables (zeros within bounds)."""
        # No explicit starting point in SIF file, use zeros
        return jnp.zeros(2 * self.n_airports)

    def bounds(self):
        """Get variable bounds."""
        # All variables bounded in [-10, 10]
        lower = jnp.full(2 * self.n_airports, -10.0)
        upper = jnp.full(2 * self.n_airports, 10.0)
        return (lower, upper)

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From comment in SIF file
        return jnp.array(47952.695811)
