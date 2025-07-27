import jax.numpy as jnp

from ..._problem import AbstractNonlinearEquations


class HATFLDG(AbstractNonlinearEquations):
    """A test problem from the OPTIMA user manual.

    Source:
    "The OPTIMA user manual (issue No.8, p. 49)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.

    classification NOR2-AY-25-25
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args):
        del args
        n = len(y)
        residuals = jnp.zeros(n)

        # First equation: x[1] - x[13] + 1 - x[1]*x[2] = 0
        residuals = residuals.at[0].set(y[0] - y[12] + 1.0 - y[0] * y[1])

        # Middle equations: x[i]*(x[i-1] - x[i+1]) + x[i] - x[13] + 1 = 0
        for i in range(1, n - 1):
            residuals = residuals.at[i].set(
                y[i] * (y[i - 1] - y[i + 1]) + y[i] - y[12] + 1.0
            )

        # Last equation: x[n] - x[13] + 1 + x[n-1]*x[n] = 0
        residuals = residuals.at[n - 1].set(
            y[n - 1] - y[12] + 1.0 + y[n - 2] * y[n - 1]
        )

        return residuals

    @property
    def y0(self):
        # Initial point from AMPL model
        return jnp.ones(25)

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
