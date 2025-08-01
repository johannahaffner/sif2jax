import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class HATFLDENE(AbstractNonlinearEquations):
    """An exponential fitting test problem from the OPTIMA user manual.

    Nonlinear-equations version of HATFLDE.

    TODO: Human review needed
    Attempts made: Multiple sign conventions tried
    Suspected issues: PyCUTEst may have different NLE convention than SIF file
    Resources needed: Understanding of how PyCUTEst interprets NLE problems
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

        # Data values from SIF file
        z_values = jnp.array(
            [
                1.561,
                1.473,
                1.391,
                1.313,
                1.239,
                1.169,
                1.103,
                1.04,
                0.981,
                0.925,
                0.8721,
                0.8221,
                0.7748,
                0.73,
                0.6877,
                0.6477,
                0.6099,
                0.5741,
                0.5403,
                0.5084,
                0.4782,
            ]
        )

        # Compute residuals for each data point
        # From SIF: G(I) = B(I) - A(I) with constant -Z(I)
        # B(I) = exp(t * x3), A(I) = x1 * exp(t * x2) with coefficient -1.0
        # For nonlinear equations, pycutest returns negative of optimization formulation
        # The residual is: -x1 * exp(t * x2) + exp(t * x3) - z
        residuals = -x1 * jnp.exp(t_values * x2) + jnp.exp(t_values * x3) - z_values

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
