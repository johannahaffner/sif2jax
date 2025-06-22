import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class DIXMAANA1(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version A1).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It's a variant of DIXMAANA with beta=0, so elements/groups
    of type 2 are removed.

    The objective function includes quadratic terms, quartic terms, and bilinear terms.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.
    update Nick Gould, August 2022, to remove beta=0 terms.

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 3000  # Default dimension

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        # Note: Beta is zero for DIXMAANA1, so not used
        gamma = 0.125
        delta = 0.125

        # Note: We don't use the i_vals directly in this implementation

        # Compute the first term (type 1): sum(alpha * (x_i)^2)
        term1 = alpha * jnp.sum(y**2)

        # Compute the third term (type 3): sum(gamma * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        term3 = gamma * jnp.sum((y[indices1] ** 2) * (y[indices2] ** 4))

        # Compute the fourth term (type 4): sum(delta * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        term4 = delta * jnp.sum(y[indices1] * y[indices2])

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
