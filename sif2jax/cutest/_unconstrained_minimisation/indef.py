import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed to verify the implementation matches the problem definition
class INDEF(AbstractUnconstrainedMinimisation):
    """A nonconvex problem which has an indefinite Hessian at the starting point.

    The parameter ALPHA controls the indefiniteness.
    Larger values of ALPHA give more indefiniteness.

    SIF input: Nick Gould, Oct 1992.
    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Other suggested values: 10, 50, 100, 1000
    alpha: float = 0.5  # Other suggested values: 1.0, 10.0, 100.0, 1000.0

    def objective(self, y, args):
        """Compute the objective function value.

        The objective consists of:
        1. Linear sum of all variables
        2. ALPHA * Sum of cosines of linear combinations of variables
        """
        # Linear sum of all variables (not squared)
        linear_sum = jnp.sum(y)

        # Calculate the COS terms for indices 2 to n-1
        # Each COS term is: ALPHA * cos(2*x_i - x_1 - x_n)
        cos_sum = 0.0
        if self.n > 2:
            # Get indices 1 to n-2 (corresponding to i=2 to i=n-1 in 1-indexed SIF)
            indices = jnp.arange(1, self.n - 1)
            # Calculate the arguments to the cosine functions
            cos_args = 2 * y[indices] - y[0] - y[self.n - 1]
            # Sum up alpha * cos(arg) for each argument
            cos_sum = jnp.sum(self.alpha * jnp.cos(cos_args))

        return linear_sum + cos_sum

    def y0(self):
        """Initial point with x_i = i/(n+1)."""
        indices = jnp.arange(1, self.n + 1)
        return indices / (self.n + 1)

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The solution is not specified in the SIF file."""
        # Since there's no specified solution, we'll return None
        # For dimensions smaller than 3, the expected result should be all zeros
        # (as only the l2 term would exist, which is minimized at zero)
        return None

    def expected_objective_value(self):
        """The solution value is not specified in the SIF file."""
        return None
