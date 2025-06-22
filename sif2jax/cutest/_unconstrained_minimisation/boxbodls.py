import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BOXBODLS(AbstractUnconstrainedMinimisation):
    """NIST Data fitting problem BOXBOD.

    Fit: y = b1*(1-exp[-b2*x]) + e

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Box, G. P., W. G. Hunter, and J. S. Hunter (1978).
    Statistics for Experimenters, New York, NY: Wiley, pp. 483-487.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: SUR2-MN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Problem has 2 variables
    m: int = 6  # Number of data points

    def objective(self, y, args):
        del args
        b1, b2 = y

        # Data points from the SIF file
        x_data = jnp.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        y_data = jnp.array([109.0, 149.0, 149.0, 191.0, 213.0, 224.0])

        # Model: y = b1*(1-exp[-b2*x])
        y_pred = b1 * (1.0 - jnp.exp(-b2 * x_data))

        # Compute the residuals
        residuals = y_pred - y_data

        # Sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (START1)
        return jnp.array([1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # NIST certified values from https://www.itl.nist.gov/div898/strd/nls/data/boxbod.shtml
        return jnp.array([213.80937, 0.54723])

    def expected_objective_value(self):
        # From NIST: sum of squared residuals is 1.1680088766E+03
        return jnp.array(1.1680088766e03)
