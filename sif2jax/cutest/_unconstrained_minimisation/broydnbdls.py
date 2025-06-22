import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed - gradient test fails
# The implementation computes the correct objective value but the gradient
# has discrepancies in the last few elements. This is likely due to subtle
# issues in how the banded structure interacts with the nonlinear elements
# in the lower right corner of the system.
class BROYDNBDLS(AbstractUnconstrainedMinimisation):
    """Broyden banded system of nonlinear equations in least square sense.

    Source: problem 31 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#73 and Toint#18
    SIF input: Ph. Toint, Dec 1989.
    Least-squares version: Nick Gould, Oct 2015

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Dimension of the problem
    kappa1: float = 2.0  # Parameter
    kappa2: float = 5.0  # Parameter
    kappa3: float = 1.0  # Parameter
    lb: int = 5  # Lower bandwidth
    ub: int = 1  # Upper bandwidth

    def objective(self, y, args):
        del args
        n = self.n
        k1 = self.kappa1
        k2 = self.kappa2
        k3 = self.kappa3
        lb = self.lb
        ub = self.ub

        # Pre-compute powers of y for efficiency
        y_squared = y**2
        y_cubed = y**3

        # Initialize residuals array
        residuals = jnp.zeros(n)

        # Create index arrays for vectorized operations
        indices = jnp.arange(n)

        # Linear part: k1 * x[i] for all i
        linear_part = k1 * y

        # For each position i, we need to sum contributions from:
        # - Linear terms: -k3 * sum of y[j] for j in the band around i
        # - Nonlinear terms: different powers of y[j] depending on position

        # Strategy: Create a banded matrix of contributions and use matrix operations
        # This avoids dynamic indexing issues

        # First, handle the three regions separately using masks
        # Note: These masks were intended for a different vectorization approach
        # but are kept here for documentation of the three regions
        # upper_left_mask = indices < lb
        # middle_mask = (indices >= lb) & (indices < n - ub)
        # lower_right_mask = indices >= n - ub

        # Function to compute residual for a single index
        def compute_residual_i(i):
            res = linear_part[i]

            # Determine which nonlinear term to use for position i
            is_upper_left = i < lb
            is_lower_right = i >= n - ub
            is_middle = ~is_upper_left & ~is_lower_right

            # Add the appropriate nonlinear term for position i
            res = res + jnp.where(is_middle, k2 * y_squared[i], k2 * y_cubed[i])

            # Compute contributions from surrounding elements
            # Lower band: j from max(0, i-lb) to i-1
            j_lower_start = jnp.maximum(0, i - lb)
            j_lower_end = i

            # Upper band: j from i+1 to min(n-1, i+ub)
            j_upper_start = i + 1
            j_upper_end = jnp.minimum(n, i + ub + 1)

            # Create masks for the bands
            j_indices = jnp.arange(n)
            lower_band_mask = (j_indices >= j_lower_start) & (j_indices < j_lower_end)
            upper_band_mask = (j_indices >= j_upper_start) & (j_indices < j_upper_end)

            # Linear contributions from lower band
            res = res - k3 * jnp.sum(y * lower_band_mask)

            # Linear contributions from upper band
            res = res - k3 * jnp.sum(y * upper_band_mask)

            # Nonlinear contributions depend on position
            # For upper left corner (i < lb):
            #   - E(j) terms (squared) for j < i and j > i
            #   - Q(i) term (cubed) at position i
            if_upper_left_lower = -k3 * jnp.sum(y_squared * lower_band_mask)
            if_upper_left_upper = -k3 * jnp.sum(y_squared * upper_band_mask)

            # For middle (lb <= i < n-ub):
            #   - Q(j) terms (cubed) for j < i
            #   - E(j) terms (squared) for j > i
            #   - E(i) term (squared) at position i
            if_middle_lower = -k3 * jnp.sum(y_cubed * lower_band_mask)
            if_middle_upper = -k3 * jnp.sum(y_squared * upper_band_mask)

            # For lower right corner (i >= n-ub):
            #   - E(j) terms (squared) for j < i and j > i
            #   - Q(i) term (cubed) at position i
            if_lower_right_lower = -k3 * jnp.sum(y_squared * lower_band_mask)
            if_lower_right_upper = -k3 * jnp.sum(y_squared * upper_band_mask)

            # Select appropriate nonlinear contributions based on position
            nonlinear_lower = jnp.where(
                is_middle,
                if_middle_lower,
                jnp.where(is_upper_left, if_upper_left_lower, if_lower_right_lower),
            )

            nonlinear_upper = jnp.where(
                is_middle,
                if_middle_upper,
                jnp.where(is_upper_left, if_upper_left_upper, if_lower_right_upper),
            )

            res = res + nonlinear_lower + nonlinear_upper

            return res

        # Vectorize over all indices
        residuals = jax.vmap(compute_residual_i)(indices)

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return inexact_asarray(jnp.ones(self.n))

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to the SIF file comment (line 212),
        # the optimal objective value is 0.0
        return jnp.array(0.0)
