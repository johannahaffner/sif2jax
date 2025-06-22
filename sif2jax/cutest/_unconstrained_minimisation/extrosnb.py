import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class EXTROSNB(AbstractUnconstrainedMinimisation):
    """Extended Rosenbrock function (nonseparable version).

    This is a scaled variant of the Rosenbrock function.
    The function is characterized by a curved narrow valley.

    The objective function is:
    f(x) = (x_1 + 1)^2 + 100 * sum_{i=2}^{n} (x_i - x_{i-1}^2)^2

    Source: problem 21 in
    J.J. Moré, B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 1000  # Default dimension (other suggested dimensions: 5, 10, 100)

    def objective(self, y, args):
        del args

        # From SIF file:
        # Group SQ1 = (X1 + 1.0) with L2 group type → (X1 + 1.0)²
        # Group SQ(I) = X(I) + ELA(I) with scale 0.01 and L2 group type
        # where ELA(I) = -X(I-1)² from ETYPE element
        # So SQ(I) = 0.01 × (X(I) - X(I-1)²)² for I=2..N

        # First term: (X1 + 1.0)²
        term1 = (y[0] + 1.0) ** 2

        # Remaining terms: 0.01 × (X(I) - X(I-1)²)² for I=2..N
        def scaled_term(i):
            # i ranges from 1 to n-1 (0-based), corresponding to SIF I=2..N
            return 0.01 * (y[i] - y[i - 1] ** 2) ** 2

        indices = jnp.arange(1, self.n)
        term2 = jnp.sum(jax.vmap(scaled_term)(indices))

        return term1 + term2

    def y0(self):
        # Starting point from the SIF file: all variables = -1.0
        return inexact_asarray(jnp.full(self.n, -1.0))

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution has all components equal to 1
        return jnp.ones(self.n)

    def expected_objective_value(self):
        # The minimum objective value is 0.0
        return jnp.array(0.0)
