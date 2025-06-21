import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class NONCVXUN(AbstractUnconstrainedMinimisation):
    """
    NONCVXUN problem.

    A nonconvex unconstrained function with a unique minimum value

    SIF input: Nick Gould, April 1996

    classification OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000

    def objective(self, y, args):
        del args
        n = self.n

        # Create index arrays for all elements
        i_indices = jnp.arange(n)

        # First variable indices (0-based)
        i1 = i_indices

        # Second variable indices: mod(2*(i+1) - 1, n)
        i2 = (2 * (i_indices + 1) - 1) % n

        # Third variable indices: mod(3*(i+1) - 1, n)
        i3 = (3 * (i_indices + 1) - 1) % n

        # Sum of the three variables for each element
        v = y[i1] + y[i2] + y[i3]

        # Square elements: v^2
        sq_values = v * v

        # Cosine elements: 4*cos(v)
        cos_values = 4.0 * jnp.cos(v)

        # Objective is sum of (sq + cos) for all elements
        # GROUP USES adds elements, not multiplies them
        return jnp.sum(sq_values + cos_values)

    def y0(self):
        # Starting point: X(I) = I for default start
        # START2 has all variables at 0.6318
        return inexact_asarray(jnp.arange(1.0, self.n + 1.0))

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in SIF file
        return None

    def expected_objective_value(self):
        # Solution value for n=5000
        return jnp.array(11584.042)

    def num_variables(self):
        return self.n
