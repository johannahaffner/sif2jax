import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed - gradient test fails
# The implementation computes the correct objective value but the gradient
# has discrepancies in the last few elements. This is likely due to subtle
# issues in how the banded structure interacts with the nonlinear elements
# in the lower right corner of the system.
class BRYBND(AbstractUnconstrainedMinimisation):
    """Broyden banded system of nonlinear equations.

    Source: problem 31 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#73 (p. 41) and Toint#18
    SIF input: Ph. Toint, Dec 1989.

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

        # Create index array
        indices = jnp.arange(n)

        # Function to compute residual for a single index
        def compute_residual_i(i):
            # Linear part: k1 * x[i]
            res = k1 * y[i]

            # Determine region
            is_upper_left = i < lb
            is_lower_right = i >= n - ub

            # Compute band indices
            j_lower_start = jnp.maximum(0, i - lb)
            j_upper_end = jnp.minimum(n, i + ub + 1)

            # Create masks for bands
            j_indices = jnp.arange(n)
            lower_band_mask = (j_indices >= j_lower_start) & (j_indices < i)
            upper_band_mask = (j_indices > i) & (j_indices < j_upper_end)

            # Sum linear contributions from bands
            res = res - k3 * jnp.sum(y * lower_band_mask)
            res = res - k3 * jnp.sum(y * upper_band_mask)

            # The nonlinear terms are ADDED, not subtracted
            # And they come from elements E(j) and Q(j) in surrounding positions
            # not just at position i

            # For upper left corner (i < lb):
            #   - Add k2 * Q(i) at position i (cubed)
            #   - Add -k3 * E(j) for j < i and j > i (squared)
            if_upper_left = (
                k2 * y_cubed[i]
                - k3 * jnp.sum(y_squared * lower_band_mask)
                - k3 * jnp.sum(y_squared * upper_band_mask)
            )

            # For middle (lb <= i < n-ub):
            #   - Add k2 * E(i) at position i (squared)
            #   - Add -k3 * Q(j) for j < i (cubed)
            #   - Add -k3 * E(j) for j > i (squared)
            if_middle = (
                k2 * y_squared[i]
                - k3 * jnp.sum(y_cubed * lower_band_mask)
                - k3 * jnp.sum(y_squared * upper_band_mask)
            )

            # For lower right corner (i >= n-ub):
            #   - Add k2 * Q(i) at position i (cubed)
            #   - Add -k3 * E(j) for j < i and j > i (squared)
            if_lower_right = (
                k2 * y_cubed[i]
                - k3 * jnp.sum(y_squared * lower_band_mask)
                - k3 * jnp.sum(y_squared * upper_band_mask)
            )

            # Select appropriate nonlinear contributions
            nonlinear_term = jnp.where(
                is_upper_left,
                if_upper_left,
                jnp.where(is_lower_right, if_lower_right, if_middle),
            )
            res = res + nonlinear_term

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
        # According to the SIF file comment (line 213),
        # the optimal objective value is 0.0
        return jnp.array(0.0)
