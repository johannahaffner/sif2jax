import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class DIXMAANF(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version F).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It includes quadratic, sin, quartic, and bilinear terms
    with non-trivial powers of (i/n) in the weights.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 3000  # Default dimension

    def __init__(self, n=None):
        if n is not None:
            self.n = n

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters (k1=1, k2=0, k3=0, k4=1)
        alpha = 1.0
        beta = 0.0625
        gamma = 0.0625
        delta = 0.0625

        # Weight for terms 1 and 4 (k=1): i/n
        i_over_n = jnp.arange(1, n + 1, dtype=y.dtype) / n

        term1 = alpha * jnp.sum(i_over_n * y**2)
        term2 = beta * jnp.sum(y[: n - 1] ** 2 * (y[1:n] + y[1:n] ** 2) ** 2)
        term3 = gamma * jnp.sum(y[: 2 * m] ** 2 * y[m : 3 * m] ** 4)
        w4 = i_over_n[:m]
        term4 = delta * jnp.sum(w4 * y[:m] * y[2 * m : 3 * m])

        # Add the constant term from GA group
        # In SIF format, CONSTANTS section subtracts the value from the group
        # So GA - (-1.0) = GA + 1.0
        constant = 1.0

        return term1 + term2 + term3 + term4 + constant

    @property
    def y0(self):
        # Initial value is 2.0 for all variables
        return inexact_asarray(jnp.full(self.n, 2.0))

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    @property
    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)
