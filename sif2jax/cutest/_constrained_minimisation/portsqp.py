import equinox as eqx
import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class PORTSQP(AbstractConstrainedQuadraticProblem):
    """A convex quadratic program of portfolio type.

    The objective function is of the form:

        sum (i=1,n) ( x_i - ( 2 i - n ) / n )^2

    There is a single equality constraint of the form:

        sum(i=1,n) x_i = 1

    Finally, there are simple bounds:

        0 <= x_i   (i=1,n)

    SIF input: Nick Gould, June 2001

    Classification: QLR2-AN-V-1

    The problem is scalable with default n = 100000.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    _n: int = eqx.field(static=True, init=False)

    def __init__(self, n=100000):
        """Initialize the problem with a given dimension.

        Args:
            n: Number of variables (default 100000)
        """
        object.__setattr__(self, "_n", n)

    @property
    def n_var(self):
        return self._n

    @property
    def n_con(self):
        return jnp.array(1)

    @property
    def y0(self):
        # Start point: all variables at 0.5
        return jnp.full(self._n, 0.5, dtype=jnp.float64)

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        # Lower bounds: 0 for all variables
        lower = jnp.zeros(self._n, dtype=jnp.float64)
        # No upper bounds (represented as infinity)
        upper = jnp.full(self._n, jnp.inf, dtype=jnp.float64)
        return lower, upper

    def objective(self, y, args):
        """Quadratic objective: sum_i (x_i - target_i)^2"""
        # Compute target values: (2*i - n) / n for i=1..n
        # Use float64 for precision with large n
        indices = jnp.arange(1, self._n + 1, dtype=jnp.float64)
        targets = (2 * indices - self._n) / self._n
        # Vectorized computation for performance
        deviations = y - targets
        return jnp.sum(deviations**2)

    def constraint(self, y):
        """Equality constraint: sum of all variables equals 1"""
        eq_constraint = jnp.sum(y) - 1.0
        return jnp.array([eq_constraint]), None

    @property
    def expected_result(self):
        # Not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        # Not provided in SIF file
        return None
