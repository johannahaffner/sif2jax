import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BOXPOWER(AbstractUnconstrainedMinimisation):
    """Function with a "box-shaped" singular Hessian.

    Source: Nick Gould, June 2013

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 20000
    p: int = 9

    @property
    def name(self):
        return "BOXPOWER"

    def objective(self, y, args):
        del args
        n = self.n
        p = self.p

        # According to the SIF file:
        # G(1) = x[0]^2 (L2 group type)
        # G(i) = (x[0] + x[i-1] + x[n-1])^2 for i=2 to n-1 (L2 group type)
        # G(n) = x[n-1]^(2*(p+1)) (POWER group type)

        # First group: x[0]^2
        g1 = y[0] ** 2

        # Middle groups: (x[0] + x[i] + x[n-1])^2 for i=1 to n-2
        middle_terms = jnp.sum((y[0] + y[1 : n - 1] + y[n - 1]) ** 2)

        # Last group: x[n-1]^p (POWER group type)
        gn = y[n - 1] ** p

        return g1 + middle_terms + gn

    def y0(self):
        # Initial values from SIF file (all 0.99)
        return jnp.full(self.n, 0.99)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to the SIF file comment (line 86),
        # the optimal objective value is 0.0
        return jnp.array(0.0)
