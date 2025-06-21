import abc

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._problem import AbstractUnconstrainedMinimisation


# Base class for all Lanczos problems
class _AbstractLanczos(AbstractUnconstrainedMinimisation):
    """Base class for NIST Data fitting problem LANCZOS series.

    Fit: y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x) + e

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Lanczos, C. (1956).
    Applied Analysis. Englewood Cliffs, NJ: Prentice Hall, pp. 272-280.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015
    Classification: SUR2-MN-6-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Set of valid starting point IDs
    valid_ids = frozenset([0, 1])

    # Starting point ID (0 or 1)
    y0_id: int = 0

    # The y_values will be defined in the derived classes

    def __check_init__(self):
        """Validate that y0_id is a valid starting point ID."""
        if self.y0_id not in self.valid_ids:
            raise ValueError(f"y0_id must be one of {self.valid_ids}")

    @abc.abstractmethod
    def _data(self) -> Array:
        """Data values are problem-specific and should be defined in subclasses."""

    def model(self, x, params):
        """Compute the model function: b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)"""
        b1, b2, b3, b4, b5, b6 = params
        term1 = b1 * jnp.exp(-b2 * x)
        term2 = b3 * jnp.exp(-b4 * x)
        term3 = b5 * jnp.exp(-b6 * x)
        return term1 + term2 + term3

    def objective(self, y, args):
        """Compute the objective function value.

        The objective is the sum of squares of residuals between the model and the data.
        """
        # Calculate the predicted values using the model
        x_values = jnp.array(
            [
                0.00,
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
                1.00,
                1.05,
                1.10,
                1.15,
            ]
        )
        y_pred = jax.vmap(lambda x: self.model(x, y))(x_values)

        # Calculate the residuals
        y_values = self._data()
        residuals = y_pred - y_values

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    def get_start_point(self, id_val):
        """Return a starting point based on the ID."""
        start_points = [
            jnp.array([1.2, 0.3, 5.6, 5.5, 6.5, 7.6]),  # START1
            jnp.array([0.5, 0.7, 3.6, 4.2, 4.0, 6.3]),  # START2
        ]
        return start_points[id_val]

    def y0(self):
        """Initial point based on the y0_id parameter."""
        return self.get_start_point(self.y0_id)

    def args(self):
        return None

    def expected_objective_value(self):
        return None
