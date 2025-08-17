import jax.numpy as jnp

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

        # Vectorized computation: for each group OBJ(i),
        # compute the sum of sine elements
        # Group i includes sine elements from positions:
        # i, (2i-1) mod n + 1, (3i-1) mod n + 1, (5i-1) mod n + 1,
        # (7i-1) mod n + 1, (11i-1) mod n + 1

        # Create array of group indices (1 to n) using same dtype as y
        i_values = jnp.arange(1, n + 1, dtype=y.dtype)

        # Compute all position indices in vectorized form (convert to 0-based)
        # SIF formula: (k*i-1) mod n + 1, convert to 0-based by subtracting 1
        # Result: ((k*i-1) mod n + 1) - 1 = (k*i-1) mod n
        pos_i = (i_values - 1).astype(jnp.int32)  # i (0-based)
        pos_2i = ((2 * i_values - 1) % n).astype(jnp.int32)  # (2i-1) mod n
        pos_3i = ((3 * i_values - 1) % n).astype(jnp.int32)  # (3i-1) mod n
        pos_5i = ((5 * i_values - 1) % n).astype(jnp.int32)  # (5i-1) mod n
        pos_7i = ((7 * i_values - 1) % n).astype(jnp.int32)  # (7i-1) mod n
        pos_11i = ((11 * i_values - 1) % n).astype(jnp.int32)  # (11i-1) mod n

        # Gather sine values at all positions
        sine_i = sine_values[pos_i]
        sine_2i = sine_values[pos_2i]
        sine_3i = sine_values[pos_3i]
        sine_5i = sine_values[pos_5i]
        sine_7i = sine_values[pos_7i]
        sine_11i = sine_values[pos_11i]

        # Sum of sine values for each group (alpha values)
        alpha_values = sine_i + sine_2i + sine_3i + sine_5i + sine_7i + sine_11i

        # Group contributions: 0.5 * P * ALPHA^2 where P = i
        group_contributions = 0.5 * i_values * alpha_values**2

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
