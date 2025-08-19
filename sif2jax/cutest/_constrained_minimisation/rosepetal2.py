import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class ROSEPETAL2(AbstractConstrainedMinimisation):
    """ROSEPETAL2 problem.

    A reformulation of ROSEPETAL, via
    minimize sum_i i x_i
    subject to sum_i x_i^2 - s = 0
    and        s +/- 2 x_j <= r^2 - 1 (1<=j<=n)
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

    def __init__(self, n: int = 10000, r: float = 2.0):
        """Initialize the ROSEPETAL2 problem.

        Args:
            n: Number of x variables (default 10000)
            r: Constraint radius >= 1 (default 2.0)
        """
        self._n = n
        self._r = r
        self._r2 = r * r
        self._r2_minus_1 = self._r2 - 1.0

    @property
    def n(self):
        """Number of variables (n + 1 including s)."""
        return self._n + 1

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        # OBJ = sum_i i * x_i (not including s)
        x = y[: self._n]
        indices = jnp.arange(1, self._n + 1, dtype=y.dtype)
        return jnp.dot(indices, x)

    def constraint(self, y):
        """Compute the constraints."""
        x = y[: self._n]
        s = y[self._n]

        # Equality constraint: sum_i x_i^2 - s = 0
        sum_squares = jnp.sum(x * x)
        equality_constraints = jnp.array([sum_squares - s])

        # Inequality constraints in the order M(j), P(j) for each j:
        # M(j): s - 2*x_j <= r^2 - 1
        # P(j): s + 2*x_j <= r^2 - 1

        # Vectorized computation with proper interleaving
        minus_constraints = s - 2.0 * x - self._r2_minus_1
        plus_constraints = s + 2.0 * x - self._r2_minus_1

        # Interleave the constraints: M(1), P(1), M(2), P(2), ...
        inequality_constraints = jnp.empty(2 * self._n)
        inequality_constraints = inequality_constraints.at[::2].set(minus_constraints)
        inequality_constraints = inequality_constraints.at[1::2].set(plus_constraints)

        return equality_constraints, inequality_constraints

    @property
    def y0(self):
        """Initial guess."""
        # Start with x = r^2, s = n * r^2
        x0 = jnp.full(self._n, self._r2)
        s0 = jnp.array([self._n * self._r2])
        return jnp.concatenate([x0, s0])

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
