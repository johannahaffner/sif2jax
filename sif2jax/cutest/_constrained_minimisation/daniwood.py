import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class DANIWOOD(AbstractConstrainedMinimisation):
    """NIST data fitting problem DANIWOOD as a constrained problem.

    This is a revised version of the original inaccurate
    formulation of DANWOOD, with corrections provided by Abel Siqueira,
    Federal University of Parana

    Fit: y = b1*x**b2

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Daniel, C. and F. S. Wood (1980).
    Fitting Equations to Data, Second Edition.
    New York, NY: John Wiley and Sons, pp. 428-431.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015 (as DANWOOD)
              correction by Abel Siqueira, Feb 2019 (renamed DANIWOOD)

    Classification: NOR2-MN-2-6
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Data values
    x_data = jnp.array([1.309, 1.471, 1.490, 1.565, 1.611, 1.680])
    y_data = jnp.array([2.138, 3.421, 3.597, 4.340, 4.882, 5.660])

    @property
    def n(self):
        """Number of variables."""
        return 2

    @property
    def m(self):
        """Number of constraints."""
        return 6  # One constraint per data point

    def objective(self, y, args):
        """For a constrained least squares formulation, the objective is 0."""
        del args, y
        return jnp.array(0.0)

    def constraint(self, y):
        """Implement the abstract constraint method."""
        eq, ineq = self.equality_constraints(y, self.args())
        return eq, ineq

    def equality_constraints(self, y, args):
        """The constraints are the residuals: b1*x**b2 - y_i = 0"""
        del args
        b1, b2 = y

        # Model predictions: b1 * x**b2
        predictions = b1 * self.x_data**b2

        # Residuals
        residuals = predictions - self.y_data

        return residuals, None

    def y0(self):
        """Initial guess."""
        # Using START1 from SIF file
        return jnp.array([1.0, 5.0])

    def args(self):
        """No additional arguments."""
        return None

    def bounds(self):
        """Variable bounds."""
        return jnp.full(self.n, -jnp.inf), jnp.full(self.n, jnp.inf)

    def expected_result(self):
        """Expected optimal solution.

        From NIST reference:
        b1 = 0.76886226
        b2 = 3.8604055
        """
        return jnp.array([0.76886226, 3.8604055])

    def expected_objective_value(self):
        """Expected optimal objective value (0 for constrained formulation)."""
        return jnp.array(0.0)
