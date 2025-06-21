import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HATFLDD(AbstractUnconstrainedMinimisation):
    """An exponential fitting test problem from the OPTIMA user manual.

    Source:
    "The OPTIMA user manual (issue No.8, p. 35)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.

    Classification: SUR2-AN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # Time points
        t_values = jnp.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9])

        # Data values
        z_values = jnp.array(
            [1.751, 1.561, 1.391, 1.239, 1.103, 0.981, 0.925, 0.8721, 0.8221, 0.7748]
        )

        # Compute model values for each data point
        # The model is x1 * exp(t * x2) - exp(t * x3)
        model_values = x1 * jnp.exp(t_values * x2) - jnp.exp(t_values * x3)

        # Compute residuals
        residuals = model_values - z_values

        # Objective function is sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial point from SIF file (lines 75-77)
        return jnp.array([1.0, -1.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # Not provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to SIF file (line 120), the optimal objective value is 6.615114D-08
        return jnp.array(6.615114e-08)
