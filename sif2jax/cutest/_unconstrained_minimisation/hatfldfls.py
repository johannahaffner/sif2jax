import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HATFLDFLS(AbstractUnconstrainedMinimisation):
    """A test problem from the OPTIMA user manual.

    Least-squares version of HATFLDF.

    Source:
    "The OPTIMA user manual (issue No.8, p. 47)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.
    Least-squares version of HATFLDF.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-AN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # Constants from SIF file (lines 40-42)
        targets = jnp.array([0.032, 0.056, 0.099])

        # Parameters for the element uses
        t_values = jnp.array([1.0, 2.0, 3.0])

        # Compute model values for each data point
        # The model is x1 + x2 * exp(t * x3)
        model_values = x1 + x2 * jnp.exp(t_values * x3)

        # Compute residuals
        residuals = model_values - targets

        # Objective function is sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial point from SIF file (line 50)
        return jnp.full(3, 0.1)

    def args(self):
        return None

    def expected_result(self):
        # Not provided in the SIF file
        return None

    def expected_objective_value(self):
        # Not provided in the SIF file
        return None
