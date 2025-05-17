import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# A base class for both LSC problems
# TODO: Human review needed to verify the implementation matches the problem definition
class _LSCBase(AbstractUnconstrainedMinimisation, strict=True):
    """Fit a circle to a set of 2D points.

    Source: Problem from the SciPy cookbook
    http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    SIF input: Nick Gould, Nov 2016
    Classification: SUR2-MN-3-0
    """

    # Set of valid starting point IDs
    valid_ids = frozenset([0, 1])

    # Starting point ID (0 or 1)
    y0_id: int = 0

    # Subclasses will define x_points and y_points

    def __check_init__(self):
        """Validate that y0_id is a valid starting point ID."""
        if self.y0_id not in self.valid_ids:
            raise ValueError(f"y0_id must be one of {self.valid_ids}")

    def objective(self, y, args):
        """Compute the objective function value.

        The residuals are the differences between the distances of each point
        to the center (x, y) and the radius r. The objective is the sum of
        squared residuals.
        """
        x, y_center, r = y

        # Calculate the squared distances from each point to the center
        dx = self.x_points - x
        dy = self.y_points - y_center

        # Calculate the distances (Euclidean norm)
        distances = jnp.sqrt(dx**2 + dy**2)

        # Calculate the residuals (distance - radius)
        residuals = distances - r

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The solution is not specified in the SIF file."""
        return None

    def expected_objective_value(self):
        """The solution value is not specified in the SIF file."""
        return None


# TODO: Human review needed to verify the implementation matches the problem definition
class LSC1LS(_LSCBase, strict=True):
    """Fit a circle to a set of 2D points: case 1, data points surround the circle.

    Source: Problem from the SciPy cookbook
    http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    SIF input: Nick Gould, Nov 2016
    Classification: SUR2-MN-3-0
    """

    # Data points from the SIF file (lines 33-44)
    x_points: jnp.ndarray = jnp.array([9.0, 35.0, -13.0, 10.0, 23.0, 0.0])
    y_points: jnp.ndarray = jnp.array([34.0, 10.0, 6.0, -14.0, 27.0, -10.0])

    def y0(self):
        """Get the starting point based on the y0_id."""
        if self.y0_id == 0:
            # START1 (lines 64-66)
            return jnp.array([105.0, 96.0, 230.0])
        elif self.y0_id == 1:
            # START2 (lines 70-72)
            return jnp.array([10.5, 9.6, 23.0])
        else:
            assert False, f"y0_id must be 0 or 1, got {self.y0_id}"


# TODO: Human review needed to verify the implementation matches the problem definition
class LSC2LS(_LSCBase, strict=True):
    """Fit a circle to a set of 2D points: case 2, data points in a small arc.

    Source: Problem from the SciPy cookbook
    http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    SIF input: Nick Gould, Nov 2016
    Classification: SUR2-MN-3-0
    """

    # Data points from the SIF file (lines 33-44)
    x_points: jnp.ndarray = jnp.array([36.0, 36.0, 19.0, 18.0, 33.0, 26.0])
    y_points: jnp.ndarray = jnp.array([14.0, 10.0, 28.0, 31.0, 18.0, 26.0])

    def y0(self):
        """Get the starting point based on the y0_id."""
        if self.y0_id == 0:
            # START1 (lines 64-66)
            return jnp.array([98.0, 36.0, 270.0])
        elif self.y0_id == 1:
            # START2 (lines 70-72)
            return jnp.array([9.8, 3.6, 27.0])
        else:
            assert False, f"y0_id must be 0 or 1, got {self.y0_id}"
