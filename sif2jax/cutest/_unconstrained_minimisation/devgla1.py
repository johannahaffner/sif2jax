import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class DEVGLA1(AbstractUnconstrainedMinimisation):
    """DeVilliers-Glasser problem 1 from the SCIPY global optimization benchmark.

    This problem involves fitting a model of the form:
    y = x₁ * x₂^t * sin(t * x₃ + x₄) + e

    to a set of data points (t, y) where t ranges from 0.1 to 2.4 in steps of 0.1
    and the y values are pre-computed based on specific parameter values.

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould, Jan 2020

    Classification: SUR2-MN-4-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 4  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2, x3, x4 = y

        # Generate time points t from 0.0 to 2.3 with step 0.1 (24 points total)
        t_values = jnp.arange(0.0, 2.4, 0.1)

        # Compute the true data points (computed from SIF file with exact parameters)
        true_values = jnp.array(
            [
                59.052474,
                54.425184,
                44.044493,
                28.575487,
                9.236288,
                -12.287936,
                -33.983363,
                -53.687835,
                -69.297789,
                -78.982793,
                -81.386703,
                -75.794406,
                -62.245213,
                -41.578070,
                -15.399602,
                14.026818,
                43.965014,
                71.449307,
                93.566643,
                107.751833,
                112.067181,
                105.437460,
                87.813812,
                60.245495,
            ]
        )

        # Compute the model predictions
        model_pred = self._model(x1, x2, x3, x4, t_values)

        # Calculate residuals
        residuals = model_pred - true_values

        # Sum of squared residuals
        return jnp.sum(residuals**2)

    def _model(self, x1, x2, x3, x4, t_values):
        """Model function: x₁ * x₂^t * sin(t * x₃ + x₄)"""
        # Compute x₂^t for all t values
        x2_power_t = x2**t_values

        # Compute sin(t * x₃ + x₄) for all t values
        sin_term = jnp.sin(t_values * x3 + x4)

        # Compute the final model prediction
        return x1 * x2_power_t * sin_term

    def y0(self):
        # Initial values from SIF file
        return inexact_asarray(jnp.array([2.0, 2.0, 2.0, 2.0]))

    def args(self):
        return None

    def expected_result(self):
        # True parameter values that generate the data
        # These values aren't explicitly given in the SIF file,
        # but can be derived from the data generation process
        return jnp.array([60.137, 1.371, 3.112, 1.761])

    def expected_objective_value(self):
        # For a perfect fit, the objective value would be 0
        return jnp.array(0.0)
