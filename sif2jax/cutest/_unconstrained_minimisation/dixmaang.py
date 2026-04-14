import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class DIXMAANG(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version G).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It includes quadratic, sin, quartic, and bilinear terms
    with non-trivial powers of (i/n) in the weights and increased parameter values.

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

        # Problem parameters
        alpha = 1.0
        beta = 0.125
        gamma = 0.125
        delta = 0.125

        # Powers for each group
        k1 = 1  # Power for group 1
        k2 = 0  # Power for group 2
        k3 = 0  # Power for group 3
        k4 = 1  # Power for group 4

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the 1st term (type 1): sum(alpha * (i/n)^k1 * (x_i)^2)
        term1 = alpha * jnp.sum(
            ((inexact_asarray(jnp.arange(1, n + 1)) / inexact_asarray(n)) ** k1)
            * (y**2)
        )

        # Compute the 2nd term (type 2):
        # sum(beta * (i/n)^k2 * x_i^2 * (x_{i+1} + x_{i+1}^2)^2)
        # for i from 1 to n-1
        w2 = (inexact_asarray(jnp.arange(1, n)) / inexact_asarray(n)) ** k2
        term2 = beta * jnp.sum(w2 * y[: n - 1] ** 2 * (y[1:n] + y[1:n] ** 2) ** 2)

        # Compute the 3rd term (type 3): sum(gamma * (i/n)^k3 * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        # Since we know n = 3m, indices will be in bounds for all i from 0 to 2m-1
        w3 = (inexact_asarray(jnp.arange(1, 2 * m + 1)) / inexact_asarray(n)) ** k3
        term3 = gamma * jnp.sum(w3 * y[: 2 * m] ** 2 * y[m : 3 * m] ** 4)

        # Compute the 4th term (type 4): sum(delta * (i/n)^k4 * x_i * x_{i+2m})
        # for i from 1 to m
        # Since we know n = 3m, indices will be exactly at the boundary
        w4 = (inexact_asarray(jnp.arange(1, m + 1)) / inexact_asarray(n)) ** k4
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
