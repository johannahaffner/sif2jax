import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HATFLDFL(AbstractUnconstrainedMinimisation):
    """Fletcher's variation of a test problem (HATFLDF) from the OPTIMA user manual.

    Monotonic paths to the solution from the initial point move to infinity and back.

    Source:
    "The OPTIMA user manual (issue No.8, p. 47)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990, mods Nick Gould, August 2008

    Nonlinear least-squares variant

    Classification: SUR2-AN-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # Constants from SIF file (lines 43-45)
        targets = jnp.array([0.032, 0.056, 0.099])

        # Parameters for the element uses
        t_values = jnp.array([1.0, 2.0, 3.0])

        # Compute model values for each data point
        # The model uses x1 + x2 * x3^t for each t in t_values
        # Note: XPEXP in this case is defined differently from HATFLDD and HATFLDE
        # Here, it's specifically x * y^t where t is an integer, not an exponential
        model_values = x1 + x2 * jnp.power(x3, t_values)

        # Compute residuals
        residuals = model_values - targets

        # Objective function is sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial point from SIF file (the "nasty" starting point, lines 55-57)
        return jnp.array([1.2, -1.2, 0.98])

    def args(self):
        return None

    def expected_result(self):
        # Not provided in the SIF file
        return None

    def expected_objective_value(self):
        # Not provided in the SIF file
        return None
