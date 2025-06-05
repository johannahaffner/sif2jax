import jax
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class EXPFIT(AbstractUnconstrainedMinimisation):
    """A simple exponential fit problem.

    This problem involves fitting an exponential function of the form
    f(x) = ALPHA * exp(BETA * x) to 10 data points.

    Source: Problem 8 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-2-0
    """

    def objective(self, y, args):
        del args

        # Extract variables
        alpha, beta = y

        # From AMPL model: sum {i in 1..p} (alpha*exp(i*h*beta)-i*h)^2
        # where p=10, h=0.25

        # Define the data points
        p = 10
        h = 0.25

        def compute_residual(i):
            # i is 1-indexed in AMPL
            # Residual: (alpha*exp(i*h*beta) - i*h)^2
            model_value = alpha * jnp.exp(i * h * beta)
            target_value = i * h
            return (model_value - target_value) ** 2

        # Compute residuals for all data points (i = 1 to 10)
        indices = jnp.arange(1, p + 1)
        residuals = jax.vmap(compute_residual)(indices)

        # Sum of squared residuals
        return jnp.sum(residuals)

    def y0(self):
        # AMPL model has no initial values specified, so variables start at 0.0
        return jnp.array([0.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not specified in the SIF file
        return None

    def expected_objective_value(self):
        return jnp.array(8.7945855171)
