import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SBRYBND(AbstractUnconstrainedMinimisation):
    """Scaled Broyden banded system of nonlinear equations,
    considered in the least square sense.

    This is a scaled version of BRYBND with exponential scaling factors
    applied to each equation.
    The problem forms a banded system with bandwidth parameters LB=5 and UB=1.

    Source: problem 31 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#73 (p. 41) and Toint#18

    SIF input: Ph. Toint and Nick Gould, Nov 1997.

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 5000

    # Problem parameters from SIF
    kappa1: float = 2.0
    kappa2: float = 5.0
    kappa3: float = 1.0
    lb: int = 5  # Lower bandwidth
    ub: int = 1  # Upper bandwidth
    scal: float = 12.0  # Scaling exponent range

    def __init__(self, n: int = 5000):
        """Initialize SBRYBND problem.

        Args:
            n: Number of variables (default 5000, must be >= 7 for LB+1+UB <= N)
        """
        if n < self.lb + 1 + self.ub:
            raise ValueError(
                f"n must be >= {self.lb + 1 + self.ub} for bandwidth constraints"
            )
        self.n = n

    def _compute_scales(self, n):
        """Compute exponential scaling factors for each variable."""
        # Scale factors: exp(scal * i/(n-1)) for i=0 to n-1
        i_vals = jnp.arange(n)
        ratios = i_vals / (n - 1)
        return jnp.exp(self.scal * ratios)

    def objective(self, y, args):
        """Compute sum-of-squares objective function.

        Fully vectorized implementation using matrix operations.
        """
        del args
        n = self.n
        scales = self._compute_scales(n)
        scaled_y = scales * y

        # Start with diagonal linear terms
        residuals = self.kappa1 * scaled_y

        # Create banded matrix for off-diagonal linear contributions
        # For efficiency, we manually handle the specific bandwidths lb=5, ub=1

        # Lower band contributions (offsets 1-5)
        if n > 1:
            residuals = residuals.at[1:].add(-self.kappa3 * scaled_y[:-1])  # offset 1
        if n > 2:
            residuals = residuals.at[2:].add(-self.kappa3 * scaled_y[:-2])  # offset 2
        if n > 3:
            residuals = residuals.at[3:].add(-self.kappa3 * scaled_y[:-3])  # offset 3
        if n > 4:
            residuals = residuals.at[4:].add(-self.kappa3 * scaled_y[:-4])  # offset 4
        if n > 5:
            residuals = residuals.at[5:].add(-self.kappa3 * scaled_y[:-5])  # offset 5

        # Upper band contributions (offset 1 only since ub=1)
        if n > 1:
            residuals = residuals.at[:-1].add(-self.kappa3 * scaled_y[1:])  # offset 1

        # Nonlinear contributions - depends on region
        i_vals = jnp.arange(n)
        upper_mask = i_vals < self.lb  # i < 5
        middle_mask = (i_vals >= self.lb) & (i_vals < n - self.ub - 1)  # 5 <= i < n-2
        lower_mask = i_vals >= n - self.ub - 1  # i >= n-2

        # Lower band nonlinear contributions
        if n > 1:
            # offset 1: apply to equations i >= 1
            contrib = -self.kappa3 * (scaled_y[:-1] ** 2)  # SQ for j > i (upper band)
            # For lower band in upper/lower regions: SQ, in middle: CB
            sq_mask = upper_mask[1:] | lower_mask[1:]
            cb_mask = middle_mask[1:]
            residuals = residuals.at[1:].add(jnp.where(sq_mask, contrib, 0.0))
            residuals = residuals.at[1:].add(
                jnp.where(cb_mask, -self.kappa3 * (scaled_y[:-1] ** 3), 0.0)
            )

        if n > 2:
            contrib = -self.kappa3 * (scaled_y[:-2] ** 2)
            sq_mask = upper_mask[2:] | lower_mask[2:]
            cb_mask = middle_mask[2:]
            residuals = residuals.at[2:].add(jnp.where(sq_mask, contrib, 0.0))
            residuals = residuals.at[2:].add(
                jnp.where(cb_mask, -self.kappa3 * (scaled_y[:-2] ** 3), 0.0)
            )

        if n > 3:
            contrib = -self.kappa3 * (scaled_y[:-3] ** 2)
            sq_mask = upper_mask[3:] | lower_mask[3:]
            cb_mask = middle_mask[3:]
            residuals = residuals.at[3:].add(jnp.where(sq_mask, contrib, 0.0))
            residuals = residuals.at[3:].add(
                jnp.where(cb_mask, -self.kappa3 * (scaled_y[:-3] ** 3), 0.0)
            )

        if n > 4:
            contrib = -self.kappa3 * (scaled_y[:-4] ** 2)
            sq_mask = upper_mask[4:] | lower_mask[4:]
            cb_mask = middle_mask[4:]
            residuals = residuals.at[4:].add(jnp.where(sq_mask, contrib, 0.0))
            residuals = residuals.at[4:].add(
                jnp.where(cb_mask, -self.kappa3 * (scaled_y[:-4] ** 3), 0.0)
            )

        if n > 5:
            contrib = -self.kappa3 * (scaled_y[:-5] ** 2)
            sq_mask = upper_mask[5:] | lower_mask[5:]
            cb_mask = middle_mask[5:]
            residuals = residuals.at[5:].add(jnp.where(sq_mask, contrib, 0.0))
            residuals = residuals.at[5:].add(
                jnp.where(cb_mask, -self.kappa3 * (scaled_y[:-5] ** 3), 0.0)
            )

        # Upper band nonlinear contributions (always SQ since j > i)
        if n > 1:
            residuals = residuals.at[:-1].add(-self.kappa3 * (scaled_y[1:] ** 2))

        # Diagonal nonlinear terms
        # Upper and lower regions: CB elements
        residuals = residuals + jnp.where(
            upper_mask | lower_mask, self.kappa2 * (scaled_y**3), 0.0
        )
        # Middle region: SQ elements
        residuals = residuals + jnp.where(middle_mask, self.kappa2 * (scaled_y**2), 0.0)

        return jnp.sum(residuals**2)

    @property
    def bounds(self):
        """All variables are unbounded."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def y0(self):
        """Starting point: x_i = 1/scale_i."""
        scales = self._compute_scales(self.n)
        return 1.0 / scales

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected solution not provided in SIF."""
        return None

    @property
    def expected_objective_value(self):
        """Expected minimum value is 0."""
        return jnp.array(0.0)
