import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class BOXBOD(AbstractNonlinearEquations):
    """NIST Data fitting problem BOXBOD as a nonlinear equations problem.

    NIST Data fitting problem BOXBOD given as an inconsistent set of
    nonlinear equations.

    Fit: y = b1*(1-exp[-b2*x]) + e

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Box, G. P., W. G. Hunter, and J. S. Hunter (1978).
    Statistics for Experimenters, New York, NY: Wiley, pp. 483-487.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: NOR2-MN-2-6
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data values
    x_data = jnp.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    y_data = jnp.array([109.0, 149.0, 149.0, 191.0, 213.0, 224.0])

    @property
    def n(self):
        """Number of variables."""
        return 2

    def num_residuals(self):
        """Number of residual equations."""
        return 6  # One for each data point

    def residual(self, y, args):
        """Compute the residuals.

        The residuals are:
        r_i = b1*(1-exp(-b2*x_i)) - y_i

        This matches the SIF formulation where F(I) = Y(I) and the
        model contribution is positive.
        """
        del args
        b1, b2 = y

        # Model predictions: b1 * (1 - exp(-b2 * x))
        predictions = b1 * (1.0 - jnp.exp(-b2 * self.x_data))

        # Residuals (model - data)
        residuals = predictions - self.y_data

        return residuals

    def y0(self):
        """Initial guess."""
        # Using START1 from SIF file
        return jnp.array([1.0, 1.0])

    def args(self):
        """No additional arguments."""
        return None

    def expected_result(self):
        """Expected optimal solution.

        From NIST reference:
        b1 = 213.80940889
        b2 = 0.54723748542
        """
        return jnp.array([213.80940889, 0.54723748542])

    def expected_objective_value(self):
        """Expected optimal objective value."""
        # For constrained formulation, objective is 0
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args()), None
