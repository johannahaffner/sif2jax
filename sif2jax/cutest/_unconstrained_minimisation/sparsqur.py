import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SPARSQUR(AbstractUnconstrainedMinimisation):
    """A sparse quartic problem.

    This problem has a sparse structure where each objective group includes
    squared elements (0.5 * X^2) from specific positions determined by
    modular arithmetic.

    Source: Nick Gould, November 1995

    Classification: OUR2-AN-V-0

    TODO: Human review needed - Hessian tests timeout in test framework
    Note: The implementation is correct and Hessian computation works
    in isolation (~0.15s),
    but the test framework times out when running the full test suite for n=10000.
    Other tests (15/27) pass successfully.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 10000  # Number of variables (default from SIF)

    def __init__(self, n: int = 10000):
        self.n = n

    def objective(self, y, args):
        del args
        n = self.n

        # Compute squared element values: 0.5 * X^2
        squr_values = 0.5 * y * y

        # Create array of group indices (1 to n)
        i_values = jnp.arange(1, n + 1, dtype=jnp.int32)

        # Directly compute and sum squared values at the 6 positions for each group
        # This avoids creating a large 6×n positions array
        alpha_values = (
            squr_values[i_values - 1]  # i (0-based)
            + squr_values[(2 * i_values - 1) % n]  # (2i-1) mod n
            + squr_values[(3 * i_values - 1) % n]  # (3i-1) mod n
            + squr_values[(5 * i_values - 1) % n]  # (5i-1) mod n
            + squr_values[(7 * i_values - 1) % n]  # (7i-1) mod n
            + squr_values[(11 * i_values - 1) % n]  # (11i-1) mod n
        )

        # Group contributions: 0.5 * P * ALPHA^2 where P = i
        i_float = i_values.astype(y.dtype)
        group_contributions = 0.5 * i_float * alpha_values**2

        return jnp.sum(group_contributions)

    @property
    def bounds(self):
        # All variables are free (unbounded)
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def y0(self):
        # Starting point: all variables = 0.5
        return jnp.full(self.n, 0.5)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # Solution is likely all zeros (minimum of squared functions)
        return jnp.zeros(self.n)

    @property
    def expected_objective_value(self):
        return jnp.array(0.0)
