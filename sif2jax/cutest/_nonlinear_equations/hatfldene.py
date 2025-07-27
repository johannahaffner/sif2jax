import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class HATFLDENE(AbstractNonlinearEquations):
    """An exponential fitting test problem from the OPTIMA user manual.

    Nonlinear-equations version of HATFLDE.

    Source:
    "The OPTIMA user manual (issue No.8, p. 37)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.
    Nonlinear-equations version of HATFLDE.SIF, Nick Gould, Jan 2020.

    classification NOR2-AN-3-21
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args):
        del args
        x1, x2, x3 = y

        # Time points
        t_values = jnp.array(
            [
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
                1.05,
                1.1,
                1.15,
                1.2,
                1.25,
                1.3,
            ]
        )

        # Data values
        z_values = jnp.array(
            [
                1.133,
                1.013,
                0.903,
                0.8025,
                0.7106,
                0.6268,
                0.5504,
                0.4810,
                0.4182,
                0.3614,
                0.3102,
                0.2644,
                0.2234,
                0.1870,
                0.1548,
                0.1266,
                0.1019,
                0.0805,
                0.0621,
                0.0465,
                0.0334,
            ]
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
        # From the original HATFLDE problem, optimal objective value is 1.472239D-09
        # Since objective = sum(residuals^2), the residual norm is sqrt(1.472239e-09)
        return jnp.sqrt(jnp.array(1.472239e-09))

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
