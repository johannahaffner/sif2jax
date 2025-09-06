import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


# TODO: Human review needed - Performance issues with 100x100 grid
# Attempts made:
#   1. Initial vectorized implementation with proper dtype handling
#   2. JIT compilation with @partial(jax.jit, static_argnums=(0,))
#   3. Minimized array operations and intermediate allocations
#   4. Single-pass difference computations with reused base points
#   5. Optimized indexing with jnp.roll instead of concatenate
#   6. Mask-based boundary condition application
# Suspected issues:
#   - Tests timeout even with aggressive optimization (60+ seconds for 10,000 variables)
#   - Likely issue with SIF interpretation (similar to JNLBRNG1)
#   - May need sparse matrix representation for efficient Hessian computation
#   - Potential numerical precision differences in trigonometric weight calculations
# Resources needed:
#   - Analysis of pycutest Fortran implementation for exact algorithm
#   - Sparse matrix support in JAX for large-scale problems
#   - Better understanding of SIF GROUP USES scaling factors


class JNLBRNG2(AbstractBoundedMinimisation):
    """The quadratic journal bearing problem (with excentricity = 0.5).

    Optimized implementation with sparse structure awareness.

    Source:
    J. More' and G. Toraldo,
    "On the Solution of Large Quadratic-Programming Problems with Bound
    Constraints",
    SIAM J. on Optimization, vol 1(1), pp. 93-113, 1991.

    SIF input: Ph. Toint, Dec 1989.
    modified by Peihuang Chen, according to MINPACK-2, Apr 1992

    Classification: QBR2-AY-V-0
    """

    PT: int = 100  # Number of points along theta
    PY: int = 100  # Number of points along Y
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return self.PT * self.PY

    def objective(self, y, args):
        """Highly optimized objective using minimal operations."""
        del args

        PT, PY = self.PT, self.PY

        # Constants - use Python floats for compile-time optimization
        EX = 0.5
        LT = 2.0 * jnp.pi
        HT = LT / (PT - 1)
        HY = 20.0 / (PY - 1)
        CLINC = -EX * HT * HY
        HY_HT = HY / HT
        HT_HY = HT / HY

        # Reshape input
        x = y.reshape(PT, PY)

        # Linear terms - vectorized
        i_vals = jnp.arange(1, PT - 1, dtype=y.dtype)
        sin_coeffs = jnp.sin(i_vals * HT) * CLINC
        linear_obj = jnp.sum(sin_coeffs[:, None] * x[1 : PT - 1, 1 : PY - 1])

        # Pre-compute cosine weights once
        all_i = jnp.arange(PT, dtype=y.dtype)
        cos_vals = jnp.cos(all_i * HT)
        w_vals = (EX * cos_vals + 1.0) ** 3

        # Lambda coefficients (right triangles)
        lambda_c = (2.0 * w_vals[: PT - 1] + w_vals[1:PT]) / 6.0

        # Vectorized quadratic terms for lambda
        # Use single pass difference computation
        x_base = x[: PT - 1, : PY - 1]  # Base reference point
        dx_right = x[1:PT, : PY - 1] - x_base  # Horizontal differences
        dx_up = x[: PT - 1, 1:PY] - x_base  # Vertical differences

        lambda_obj = jnp.sum(
            lambda_c[:, None] * (HY_HT * dx_right**2 + HT_HY * dx_up**2)
        )

        # Mu coefficients (left triangles) - optimized indexing
        w_left = jnp.roll(w_vals[: PT - 1], 1)
        w_left = w_left.at[0].set(w_vals[0])
        mu_c = (2.0 * w_vals[1:PT] + w_left) / 6.0

        # Vectorized quadratic terms for mu
        x_base2 = x[1:PT, 1:PY]  # Different base for mu terms
        dx_left = x[: PT - 1, 1:PY] - x_base2  # Horizontal differences (reversed)
        dx_down = x[1:PT, : PY - 1] - x_base2  # Vertical differences (reversed)

        mu_obj = jnp.sum(mu_c[:, None] * (HY_HT * dx_left**2 + HT_HY * dx_down**2))

        return linear_obj + lambda_obj + mu_obj

    @property
    def y0(self):
        """Initial guess - vectorized."""
        PT, PY = self.PT, self.PY

        LT = 2.0 * jnp.pi
        HT = LT / (PT - 1)

        # Create initial array
        x0 = jnp.zeros((PT, PY))

        # Set interior points
        if PT > 2 and PY > 2:
            i_vals = jnp.arange(1, PT - 1, dtype=x0.dtype)
            sin_vals = jnp.sin(i_vals * HT)
            x0 = x0.at[1 : PT - 1, 1 : PY - 1].set(sin_vals[:, None])

        return x0.reshape(-1)

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Variable bounds - optimized."""
        PT, PY = self.PT, self.PY

        # Start with all positive
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, jnp.inf)

        # Create mask for boundary points (more efficient)
        mask_2d = jnp.ones((PT, PY), dtype=bool)
        mask_2d = mask_2d.at[0, :].set(False)
        mask_2d = mask_2d.at[PT - 1, :].set(False)
        mask_2d = mask_2d.at[1 : PT - 1, 0].set(False)
        mask_2d = mask_2d.at[1 : PT - 1, PY - 1].set(False)

        # Apply mask
        mask_flat = mask_2d.reshape(-1)
        upper = jnp.where(mask_flat, jnp.inf, 0.0)

        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file comments:
        if self.PT == 4 and self.PY == 4:
            return jnp.asarray(-0.4764000)
        elif self.PT == 10 and self.PY == 10:
            return jnp.asarray(-0.3952800)
        elif self.PT == 23 and self.PY == 23:
            return jnp.asarray(-0.4102400)
        elif self.PT == 32 and self.PY == 32:
            return jnp.asarray(-0.4124900)
        elif self.PT == 75 and self.PY == 75:
            return jnp.asarray(-0.4146600)
        elif self.PT == 100 and self.PY == 100:
            return jnp.asarray(-0.4148700)
        elif self.PT == 125 and self.PY == 125:
            return jnp.asarray(-0.4149600)
        return None
