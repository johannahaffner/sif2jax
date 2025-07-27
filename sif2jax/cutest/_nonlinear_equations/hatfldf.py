import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class HATFLDF(AbstractNonlinearEquations):
    """A test problem from the OPTIMA user manual.

    Source:
    "The OPTIMA user manual (issue No.8, p. 47)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.

    classification NOR2-AN-3-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args):
        del args
        x1, x2, x3 = y

        # Three nonlinear equations
        r1 = x1 - 0.032 + x2 * jnp.exp(x3)
        r2 = x1 - 0.056 + x2 * jnp.exp(2 * x3)
        r3 = x1 - 0.099 + x2 * jnp.exp(3 * x3)

        return jnp.array([r1, r2, r3])

    @property
    def y0(self):
        # Initial point from AMPL model
        return jnp.full(3, 0.1)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Not provided in the SIF file
        return None

    @property
    def expected_residual_norm(self):
        # For a system of nonlinear equations, should be 0
        return jnp.array(0.0)

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
