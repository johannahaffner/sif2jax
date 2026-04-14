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

        # Off-diagonal linear contributions using a single pad + slices.
        # Pad scaled_y with lb zeros on the left and ub zeros on the right,
        # then each offset k is just a slice of the padded array.
        lb = self.lb  # 5
        ub = self.ub  # 1
        k3 = self.kappa3
        padded = jnp.concatenate([
            jnp.zeros(lb, dtype=scaled_y.dtype),
            scaled_y,
            jnp.zeros(ub, dtype=scaled_y.dtype),
        ])  # length n + lb + ub

        # Lower band: offset k means neighbor at position i-k
        for k in range(1, lb + 1):
            residuals = residuals - k3 * padded[lb - k : lb - k + n]

        # Upper band: offset k means neighbor at position i+k
        for k in range(1, ub + 1):
            residuals = residuals - k3 * padded[lb + k : lb + k + n]

        # Nonlinear contributions - depends on region
        i_vals = jnp.arange(n)
        upper_mask = i_vals < lb  # i < 5
        middle_mask = (i_vals >= lb) & (i_vals < n - ub - 1)  # 5 <= i < n-2
        lower_mask = i_vals >= n - ub - 1  # i >= n-2

        # Precompute nonlinear terms for the neighbor value
        nl_sq = -k3 * scaled_y**2  # SQ element
        nl_cb = -k3 * scaled_y**3  # CB element

        # For each equation i, the nonlinear element type depends on i's region.
        # Lower band: neighbor j = i-k contributes SQ (upper/lower region) or
        # CB (middle region) based on the mask at position i.
        # Pad the neighbor terms and select per-equation.
        nl_sq_padded = jnp.concatenate([
            jnp.zeros(lb, dtype=scaled_y.dtype),
            nl_sq,
            jnp.zeros(ub, dtype=scaled_y.dtype),
        ])
        nl_cb_padded = jnp.concatenate([
            jnp.zeros(lb, dtype=scaled_y.dtype),
            nl_cb,
            jnp.zeros(ub, dtype=scaled_y.dtype),
        ])

        # Lower band nonlinear: for each offset k, select SQ or CB at each i
        sq_region = upper_mask | lower_mask
        for k in range(1, lb + 1):
            neighbor_sq = nl_sq_padded[lb - k : lb - k + n]
            neighbor_cb = nl_cb_padded[lb - k : lb - k + n]
            residuals = residuals + jnp.where(
                sq_region, neighbor_sq, neighbor_cb
            )

        # Upper band nonlinear (always SQ since j > i)
        for k in range(1, ub + 1):
            residuals = residuals + nl_sq_padded[lb + k : lb + k + n]

        # Diagonal nonlinear terms
        # Upper and lower regions: CB elements
        residuals = residuals + jnp.where(
            sq_region, self.kappa2 * scaled_y**3, 0.0
        )
        # Middle region: SQ elements
        residuals = residuals + jnp.where(
            middle_mask, self.kappa2 * scaled_y**2, 0.0
        )

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
