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

    # Set of valid starting point IDs
    valid_ids = frozenset([0, 1])

    # Starting point ID (0 or 1)
    y0_id: int = 0

    # Independent variable values (x) - same for all Lanczos problems
    x_values: jnp.ndarray = jnp.array(
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
        y_pred = jax.vmap(lambda x: self.model(x, y))(self.x_values)

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


# TODO: Human review needed to verify the implementation matches the problem definition
class LANCZOS1LS(_AbstractLanczos):
    """NIST Data fitting problem LANCZOS1.

    In LANCZOS1, the y values are artificially created using the exact model
    with known parameter values.
    """

    def _data(self):
        """Initialize y_values based on the exact model."""
        # The y values are artificially created with known parameter values
        exact_params = jnp.array([0.0951, 1.0, 0.8607, 3.0, 1.5576, 5.0])
        y = jax.vmap(lambda x: self.model(x, exact_params))(self.x_values)
        return y

    def expected_result(self):
        """The exact solution for LANCZOS1."""
        return jnp.array([0.0951, 1.0, 0.8607, 3.0, 1.5576, 5.0])


# TODO: Human review needed to verify the implementation matches the problem definition
class LANCZOS2LS(_AbstractLanczos):
    """NIST Data fitting problem LANCZOS2.

    In LANCZOS2, the y values are provided directly in the SIF file.
    """

    # Dependent variable values (y) - specific to LANCZOS2
    def _data(self):
        y = jnp.array(
            [
                2.51340,
                2.04433,
                1.66840,
                1.36642,
                1.12323,
                0.92689,
                0.76793,
                0.63888,
                0.53378,
                0.44794,
                0.37759,
                0.31974,
                0.27201,
                0.23250,
                0.19966,
                0.17227,
                0.14934,
                0.13007,
                0.11381,
                0.10004,
                0.08833,
                0.07834,
                0.06977,
                0.06239,
            ]
        )
        return y

    def expected_result(self):
        return None


# TODO: Human review needed to verify the implementation matches the problem definition
class LANCZOS3LS(_AbstractLanczos):
    """NIST Data fitting problem LANCZOS3.

    In LANCZOS3, the y values are provided directly in the SIF file.
    """

    def _data(self):
        y = jnp.array(
            [
                2.51340,
                2.04433,
                1.66840,
                1.36642,
                1.12323,
                0.92689,
                0.76793,
                0.63888,
                0.53378,
                0.44794,
                0.37759,
                0.31974,
                0.27201,
                0.23250,
                0.19966,
                0.17227,
                0.14934,
                0.13007,
                0.11381,
                0.10004,
                0.08833,
                0.07834,
                0.06977,
                0.06239,
            ]
        )
        return y

    def expected_result(self):
        return None
