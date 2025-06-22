import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class DEVGLA2(AbstractUnconstrainedMinimisation):
    """DeVilliers-Glasser problem 2 from the SCIPY global optimization benchmark.

    This problem involves fitting a model of the form:
    y = x₁ * x₂^t * tanh(t * x₃ + sin(t * x₄)) * cos(t * e^x₅) + e

    to a set of data points (t, y) where t ranges from 0.1 to 1.6 in steps of 0.1
    and the y values are pre-computed based on specific parameter values.

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould, Jan 2020

    Classification: SUR2-MN-5-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5 = y

        # Generate time points t from 0.0 to 1.5 with step 0.1 (16 points total)
        t_values = jnp.arange(0.0, 1.6, 0.1)

        # Compute the true data points (computed from SIF file with exact parameters)
        true_values = jnp.array(
            [
                0.000000,
                25.652980,
                40.985849,
                45.968797,
                44.800863,
                40.224285,
                33.489336,
                25.171629,
                15.612834,
                5.094175,
                -6.103169,
                -17.680274,
                -29.318700,
                -40.685106,
                -51.438706,
                -61.240116,
            ]
        )

        # Compute the model predictions
        model_pred = self._model(x1, x2, x3, x4, x5, t_values)

        # Calculate residuals
        residuals = model_pred - true_values

        # Sum of squared residuals
        return jnp.sum(residuals**2)

    def _model(self, x1, x2, x3, x4, x5, t_values):
        """Model function: x₁ * x₂^t * tanh(t * x₃ + sin(t * x₄)) * cos(t * e^x₅)"""
        # Compute x₂^t for all t values
        x2_power_t = x2**t_values

        # Compute sin(t * x₄) for all t values
        sin_tx4 = jnp.sin(t_values * x4)

        # Compute tanh(t * x₃ + sin(t * x₄)) for all t values
        tanh_term = jnp.tanh(t_values * x3 + sin_tx4)

        # Compute cos(t * e^x₅) for all t values
        cos_term = jnp.cos(t_values * jnp.exp(x5))

        # Compute the final model prediction
        return x1 * x2_power_t * tanh_term * cos_term

    def y0(self):
        # Initial values from SIF file
        return inexact_asarray(jnp.array([20.0, 2.0, 2.0, 2.0, 0.2]))

    def args(self):
        return None

    def expected_result(self):
        # Solution values given in the SIF file
        return jnp.array([53.81, 1.27, 3.012, 2.13, 0.507])

    def expected_objective_value(self):
        # For a perfect fit, the objective value would be 0
        return jnp.array(0.0)
