import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class POWER(AbstractUnconstrainedMinimisation):
    """
    POWER problem.

    The Power problem by Oren.

    Source:
    S.S. Oren,
    Self-scaling variable metric algorithms,
    Part II: implementation and experiments"
    Management Science 20(5):863-874, 1974.

    See also Buckley#179 (p. 83)

    SIF input: Ph. Toint, Dec 1989.

    classification OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 10000

    def objective(self, y, args):
        del args
        n = self.n

        # Single group G with L2 type
        # Elements: E(i) = x[i]^2 with weight i
        # Group value: sum_{i=1}^n i * x[i]^2
        weights = jnp.arange(1.0, n + 1.0)
        weighted_sum = jnp.sum(weights * y**2)

        # L2 group type squares the result
        return weighted_sum**2

    def y0(self):
        # Starting point: all variables at 1.0
        return inexact_asarray(jnp.ones(self.n))

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in detail in SIF file
        # But optimal is all zeros
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # Solution value is 0.0
        return jnp.array(0.0)

    def num_variables(self):
        return self.n
