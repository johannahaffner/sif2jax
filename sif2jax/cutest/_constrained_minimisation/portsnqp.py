import equinox as eqx
import jax.numpy as jnp

from ..._problem import AbstractConstrainedQuadraticProblem


class PORTSNQP(AbstractConstrainedQuadraticProblem):
    """A convex quadratic program of portfolio type (non-squared variant).

    The objective function is of the form:

        sum (i=1,n) ( x_i - ( 2 i - n ) / n )^2 - 0.5 * x_n

    There are two equality constraints:

        sum(i=1,n) x_i = 1
        sum(i=1,n) ((2i-n)/n) * x_i = 0.5

    Finally, there are simple bounds:

        0 <= x_i   (i=1,n)

    SIF input: Nick Gould, June 2001

    Classification: QLR2-AN-V-1 (but actually has 2 constraints, not 1)

    The problem is scalable with default n = 10.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    _n: int = eqx.field(static=True, init=False)

    def __init__(self, n=10):
        """Initialize the problem with a given dimension.

        Args:
            n: Number of variables (default 10, but can scale to 100000)
        """
        object.__setattr__(self, "_n", n)

    @property
    def n_var(self):
        return self._n

    @property
    def n_con(self):
        return 2  # Two equality constraints

    @property
    def y0(self):
        # Start point: all variables at 0.5
        return jnp.full(self._n, 0.5)

    @property
    def args(self):
        return None

    @property
    def bounds(self):
        # Lower bounds: 0 for all variables
        lower = jnp.zeros(self._n)
        # No upper bounds (represented as infinity)
        upper = jnp.full(self._n, jnp.inf)
        return lower, upper

    def objective(self, y, args):
        """Quadratic objective with additional scaling."""
        # Compute target values: (2*i - n) / n for i=1..n
        indices = jnp.arange(1, self._n + 1, dtype=y.dtype)
        targets = (2 * indices - self._n) / self._n
        # Main quadratic term: sum_i (x_i - target_i)^2
        deviations = y - targets
        quad_term = jnp.sum(deviations**2)

        # Additional scaling term from SIF file
        # The 'SCALE' term with -0.5 coefficient on x(N)
        scale_term = -0.5 * y[-1]

        return quad_term + scale_term

    def constraint(self, y):
        """Two equality constraints."""
        # First constraint: sum of all variables equals 1
        eq1 = jnp.sum(y) - 1.0

        # Second constraint: weighted sum equals 0.5
        indices = jnp.arange(1, self._n + 1, dtype=y.dtype)
        weights = (2 * indices - self._n) / self._n
        eq2 = jnp.dot(weights, y) - 0.5

        eq_constraints = jnp.array([eq1, eq2])
        return eq_constraints, None

    @property
    def expected_result(self):
        # Not provided in SIF file
        return None

    @property
    def expected_objective_value(self):
        # Not provided in SIF file
        return None
