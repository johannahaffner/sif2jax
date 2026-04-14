import jax.numpy as jnp
import numpy as np

from ..._problem import AbstractUnconstrainedMinimisation


class SPARSINE(AbstractUnconstrainedMinimisation):
    """A sparse problem involving sine functions.

    This problem has a sparse structure where each objective group includes
    sine elements from specific positions determined by modular arithmetic.

    Source: Nick Gould, November 1995

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Number of variables

    def __init__(self, n: int = 5000):
        self.n = n

    def objective(self, y, args):
        del args
        n = self.n

        # Compute sine of all variables
        sine_values = jnp.sin(y)

        # Modular permutation indices as numpy arrays — folded as constants
        # by XLA during tracing (no iota/modular arithmetic in the jaxpr).
        # SIF formula: (k*i-1) mod n, for k in {2, 3, 5, 7, 11}
        i = np.arange(1, n + 1)
        perm_indices = np.stack([
            (2 * i - 1) % n,
            (3 * i - 1) % n,
            (5 * i - 1) % n,
            (7 * i - 1) % n,
            (11 * i - 1) % n,
        ])  # (5, n)

        # Sum of sine values for each group (alpha values)
        # The identity permutation (pos_i = arange(n)) is just sine_values
        alpha_values = sine_values + sine_values[perm_indices].sum(axis=0)

        # Group contributions: 0.5 * i * alpha^2 where i = 1..n
        # Use jnp.arange for the weight so it's a traced iota, not a constant
        group_contributions = (
            0.5 * jnp.arange(1, n + 1, dtype=y.dtype) * alpha_values**2
        )

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
        # Solution is likely all zeros (minimum of sine functions)
        return jnp.zeros(self.n)

    @property
    def expected_objective_value(self):
        return jnp.array(0.0)
