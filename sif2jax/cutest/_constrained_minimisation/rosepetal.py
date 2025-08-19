import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class ROSEPETAL(AbstractConstrainedMinimisation):
    """ROSEPETAL problem.

    An amusement: minimize sum_i i x_i
    subject to (x_j +/- 1)^2 + sum_{i≠j} x_i^2 <= r^2 (1<=j<=n)
    where r >= 1

    SIF input: Nick Gould, Feb 2014

    Classification: LQR2-AN-V-V
    """

    _n: int
    _r: float
    _r2: float
    _r2_minus_1: float
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n: int = 1000, r: float = 2.0):
        """Initialize the ROSEPETAL problem.

        Args:
            n: Number of variables (default 1000)
            r: Constraint radius >= 1 (default 2.0)
        """
        self._n = n
        self._r = r
        self._r2 = r * r
        self._r2_minus_1 = self._r2 - 1.0

    @property
    def n(self):
        """Number of variables."""
        return self._n

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        # OBJ = sum_i i * x_i
        indices = jnp.arange(1, self._n + 1, dtype=y.dtype)
        return jnp.dot(indices, y)

    def constraint(self, y):
        """Compute the constraints."""
        # No equality constraints
        equality_constraints = None

        # Inequality constraints: (x_j +/- 1)^2 + sum_{i≠j} x_i^2 <= r^2
        # Rewritten as: x_j^2 +/- 2*x_j + 1 + sum_{i≠j} x_i^2 <= r^2
        # Which is: sum_i x_i^2 +/- 2*x_j + 1 <= r^2
        # Or: sum_i x_i^2 +/- 2*x_j <= r^2 - 1

        sum_squares = jnp.sum(y * y)

        # For each j, we have two constraints in the order M(j), P(j):
        # M(j): sum_i x_i^2 - 2*x_j <= r^2 - 1
        # P(j): sum_i x_i^2 + 2*x_j <= r^2 - 1

        # Vectorized computation with proper interleaving
        minus_constraints = sum_squares - 2.0 * y - self._r2_minus_1
        plus_constraints = sum_squares + 2.0 * y - self._r2_minus_1

        # Interleave the constraints: M(1), P(1), M(2), P(2), ...
        inequality_constraints = jnp.empty(2 * self._n)
        inequality_constraints = inequality_constraints.at[::2].set(minus_constraints)
        inequality_constraints = inequality_constraints.at[1::2].set(plus_constraints)

        return equality_constraints, inequality_constraints

    @property
    def y0(self):
        """Initial guess."""
        # Start with x = r^2
        return jnp.full(self._n, self._r2)

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Bounds on variables."""
        # All variables are free
        return None, None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None  # Not provided in SIF

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return None  # Not provided in SIF
