import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class DIXMAANE1(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version E1).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It's a variant of DIXMAANE with beta=0, so elements/groups
    of type 2 are removed.

    The objective function includes quadratic terms, quartic terms, and bilinear terms
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

        # Problem parameters
        alpha = 1.0
        # Beta is zero for DIXMAANE1
        gamma = 0.125
        delta = 0.125

        # Powers for each group
        k1 = 1  # Power for group 1
        # k2 not used since beta=0
        k3 = 0  # Power for group 3
        k4 = 1  # Power for group 4

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the first term (type 1): sum(alpha * (i/n)^k1 * (x_i)^2)
        term1 = alpha * jnp.sum(
            ((inexact_asarray(jnp.arange(1, n + 1)) / n) ** k1) * (y**2)
        )

        # Compute the third term (type 3): sum(gamma * (i/n)^k3 * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        # Since we know n = 3m, indices2 will be in bounds for all i from 0 to 2m-1
        term3 = gamma * jnp.sum(
            ((indices1 + 1) / n) ** k3 * (y[indices1] ** 2) * (y[indices2] ** 4)
        )

        # Compute the fourth term (type 4): sum(delta * (i/n)^k4 * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        # Since we know n = 3m, indices2 will be exactly at the boundary
        term4 = delta * jnp.sum(((indices1 + 1) / n) ** k4 * y[indices1] * y[indices2])

        # Add the constant term from GA group
        # In SIF format, CONSTANTS section subtracts the value from the group
        # So GA - (-1.0) = GA + 1.0
        constant = 1.0

        return term1 + term3 + term4 + constant

    def y0(self):
        # Initial value is 2.0 for all variables
        return inexact_asarray(jnp.full(self.n, 2.0))

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)
