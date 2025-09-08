import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class MGH09LS(AbstractUnconstrainedMinimisation):
    """NIST Data fitting problem MGH09 (least squares version).

    Fit model: y = b1*(x**2+x*b2) / (x**2+x*b3+b4) + e

    Source:  Problem from the NIST nonlinear regression test set
      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Kowalik, J.S., and M. R. Osborne, (1978).
      Methods for Unconstrained Optimization Problems.
      New York, NY:  Elsevier North-Holland.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: SUR2-MN-4-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    # Data points (x, y)
    data_x = jnp.array(
        [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625]
    )
    data_y = jnp.array(
        [
            0.1957,
            0.1947,
            0.1735,
            0.160,
            0.0844,
            0.0627,
            0.0456,
            0.0342,
            0.0323,
            0.0235,
            0.0246,
        ]
    )

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(4)

    @property
    def y0(self):
        """Initial guess."""
        if self.y0_iD == 0:
            # START1
            return jnp.array([25.0, 39.0, 41.5, 39.0])
        else:
            # START2
            return jnp.array([0.25, 0.39, 0.415, 0.39])

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the sum of squares objective function."""
        del args  # Not used

        b1, b2, b3, b4 = y[0], y[1], y[2], y[3]

        # Model: y = b1*(x**2+x*b2) / (x**2+x*b3+b4)
        x2 = self.data_x * self.data_x
        numerator = b1 * (x2 + self.data_x * b2)
        denominator = x2 + self.data_x * b3 + b4
        model_values = numerator / denominator

        # Sum of squared residuals
        residuals = self.data_y - model_values
        return jnp.sum(residuals**2)

    @property
    def expected_result(self):
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    @property
    def expected_objective_value(self):
        """Expected value of the objective at the solution."""
        # The SIF file doesn't provide a solution
        return None
