import abc

import jax.numpy as jnp
from jaxtyping import Array

from ..._problem import AbstractUnconstrainedMinimisation


# A base class for both LSC problems
# TODO: Human review needed to verify the implementation matches the problem definition
class _AbstractLSC(AbstractUnconstrainedMinimisation):
    """Fit a circle to a set of 2D points.

    Source: Problem from the SciPy cookbook
    http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

    SIF input: Nick Gould, Nov 2016
    Classification: SUR2-MN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Set of valid starting point IDs
    valid_ids = frozenset([0, 1])

    # Starting point ID (0 or 1)
    y0_id: int = 0

    # Subclasses will define x_points and y_points

    def __check_init__(self):
        """Validate that y0_id is a valid starting point ID."""
        if self.y0_id not in self.valid_ids:
            raise ValueError(f"y0_id must be one of {self.valid_ids}")

    @abc.abstractmethod
    def _data(self) -> tuple[Array, Array]:
        """Abstract method to be implemented by subclasses to provide data points."""

    def objective(self, y, args):
        """Compute the objective function value.

        The residuals are the differences between the distances of each point
        to the center (x, y) and the radius r. The objective is the sum of
        squared residuals.
        """
        x, y_center, r = y

        # Calculate the squared distances from each point to the center
        x_points, y_points = self._data()
        dx = x_points - x
        dy = y_points - y_center

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
