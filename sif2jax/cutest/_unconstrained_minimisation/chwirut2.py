import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO human review required - needs extra data?
class CHWIRUT2(AbstractUnconstrainedMinimisation):
    """NIST data fitting problem CHWIRUT2.

    Fit: y = exp[-b1*x]/(b2+b3*x)

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Chwirut, D., NIST (197?).
    Ultrasonic Reference Block Study.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: NOR2-MN-3-54
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # TODO: This problem has 54 data points defined in the SIF file
    # However, it should ideally use external data file for consistency with NIST

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
            "CHWIRUT2 requires implementation with 54 data points"
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
        b1 = 0.16657666
        b2 = 0.0051653291
        b3 = 0.012150007
        """
        return jnp.array([0.16657666, 0.0051653291, 0.012150007])

    def expected_objective_value(self):
        """Expected optimal objective value.

        From NIST: Residual Sum of Squares = 513.04802
        """
        return jnp.array(513.04802)
