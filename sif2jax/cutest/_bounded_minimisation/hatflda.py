import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HATFLDA(AbstractBoundedMinimisation):
    """A test problem from the OPTIMA user manual.

    Source:
    "The OPTIMA user manual (issue No.8, p. 12)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.

    classification SBR2-AN-4-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args

        # Objective is (x[1]-1)^2 + sum_{i=2..N} (x[i-1] - sqrt(x[i]))^2
        obj = (y[0] - 1.0) ** 2

        # Add the sum terms
        for i in range(1, len(y)):
            obj += (y[i - 1] - jnp.sqrt(y[i])) ** 2

        return obj

    @property
    def y0(self):
        # Initial point from AMPL model
        return jnp.full(4, 0.1)

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        # Lower bounds from AMPL model
        lower = jnp.full(4, 0.0000001)
        upper = jnp.full(4, jnp.inf)
        return lower, upper

    @property
    def expected_result(self):
        # Not provided in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # Not provided in the SIF file
        return None
