import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class HATFLDDNE(AbstractNonlinearEquations):
    """An exponential fitting test problem from the OPTIMA user manual.

    Nonlinear-equations version of HATFLDD.

    Source:
    "The OPTIMA user manual (issue No.8, p. 35)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.
    Nonlinear-equations version of HATFLDD.SIF, Nick Gould, Jan 2020.

    classification NOR2-AN-3-10
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args):
        del args
        x1, x2, x3 = y

        # Time points
        t_values = jnp.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9])

        # Data values
        z_values = jnp.array(
            [1.751, 1.561, 1.391, 1.239, 1.103, 0.981, 0.925, 0.8721, 0.8221, 0.7748]
        )

        # Compute residuals for each data point
        # The model is exp(t * x3) - x1 * exp(t * x2) + z
        residuals = jnp.exp(t_values * x3) - x1 * jnp.exp(t_values * x2) + z_values

        return residuals

    @property
    def y0(self):
        # Initial point from SIF file
        return jnp.array([1.0, -1.0, 0.0])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Not provided in the SIF file
        return None

    @property
    def expected_residual_norm(self):
        # From the original HATFLDD problem, optimal objective value is 6.615114e-08
        # Since objective = sum(residuals^2), the residual norm is sqrt(6.615114e-08)
        return jnp.sqrt(jnp.array(6.615114e-08))

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def expected_objective_value(self):
        """For nonlinear equations, objective is always zero."""
        return jnp.array(0.0)

    @property
    def bounds(self):
        """No bounds for this problem."""
        return None
