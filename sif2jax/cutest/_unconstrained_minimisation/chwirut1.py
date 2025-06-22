import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO human review required - needs extra data?
class CHWIRUT1(AbstractUnconstrainedMinimisation):
    """NIST data fitting problem CHWIRUT1.

    Fit: y = exp[-b1*x]/(b2+b3*x)

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Chwirut, D., NIST (197?).
    Ultrasonic Reference Block Study.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: NOR2-MN-3-214
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # TODO: Full data (214 points) needs to be loaded from file
    # This problem requires external data that is not available in the SIF file

    @property
    def n(self):
        """Number of variables."""
        return 3

    def objective(self, y, args):
        """Compute the objective function.

        The objective is the sum of squared residuals:
        f(b) = sum_i (y_i - exp(-b1*x_i)/(b2+b3*x_i))^2
        """
        raise NotImplementedError(
            "CHWIRUT1 requires external data file with 214 data points"
        )

    def y0(self):
        """Initial guess."""
        # START1 from SIF file
        return jnp.array([0.1, 0.01, 0.02])

    def args(self):
        """No additional arguments."""
        return None

    def expected_result(self):
        """Expected optimal solution.

        From NIST reference:
        b1 = 0.19027970
        b2 = 0.0061314327
        b3 = 0.0105309915
        """
        return jnp.array([0.19027970, 0.0061314327, 0.0105309915])

    def expected_objective_value(self):
        """Expected optimal objective value.

        From NIST: Residual Sum of Squares = 2384.9741
        """
        return jnp.array(2384.9741)
