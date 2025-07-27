import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HATFLDB(AbstractBoundedMinimisation):
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
        # Same objective as HATFLDA
        # min sum_{i=1}^{n-1} (x[i] - 1)^2 where i = 1, x[i] = sqrt(x[i+1]) for i > 1
        # Rewritten as: min (x[1] - 1)^2 + sum_{i=2}^n (x[i-1] - sqrt(x[i]))^2
        obj = (y[0] - 1.0) ** 2
        for i in range(1, len(y)):
            obj += (y[i - 1] - jnp.sqrt(y[i])) ** 2
        return obj

    @property
    def y0(self):
        # Initial point from SIF: all 0.1
        return jnp.full(4, 0.1)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        # From SIF file: 5.57281D-03
        return jnp.array(5.57281e-03)

    @property
    def bounds(self):
        # Lower bound: all >= 0.0000001
        # Upper bound: x[2] <= 0.8, others unbounded
        lower = jnp.full(4, 0.0000001)
        upper = jnp.array([jnp.inf, 0.8, jnp.inf, jnp.inf])
        return lower, upper
