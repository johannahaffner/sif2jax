import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractBoundedMinimisation


# TODO: Human review needed
# Attempts made: Multiple optimization attempts with vectorization
# Suspected issues: Objective doesn't match pycutest reference (9.66 vs 25-54)
# Resources needed: Analysis of SIF GROUP USES section


class JNLBRNG1(AbstractBoundedMinimisation):
    """The quadratic journal bearing problem (with excentricity = 0.1).

    This is a variant of the problem stated in the report quoted below.
    It corresponds to the problem as distributed in MINPACK-2.

    Source:
    J. More' and G. Toraldo,
    "On the Solution of Large Quadratic-Programming Problems with Bound
    Constraints",
    SIAM J. on Optimization, vol 1(1), pp. 93-113, 1991.

    SIF input: Ph. Toint, Dec 1989.
    modified by Peihuang Chen, according to MINPACK-2, Apr 1992

    Classification: QBR2-AY-V-0
    """

    PT: int = 4  # Number of points along theta
    PY: int = 4  # Number of points along Y
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return self.PT * self.PY

    def objective(self, y, args):
        """Ultra-optimized objective function."""
        del args

        # Use compile-time constants - ensure proper dtype consistency
        PT, PY = self.PT, self.PY

        # Pre-computed constants with proper dtype
        EX = inexact_asarray(0.1)
        LT = inexact_asarray(2.0 * jnp.pi)  # Simplified from 8*pi/4
        HT = LT / inexact_asarray(PT - 1)
        HY = inexact_asarray(20.0) / inexact_asarray(PY - 1)
        CLINC = -EX * HT * HY
        HY_HT = HY / HT
        HT_HY = HT / HY

        # Single reshape, no intermediate arrays
        x = y.reshape(PT, PY)

        # Start with linear terms if they exist
        obj = inexact_asarray(0.0)
        if PT > 2 and PY > 2:
            i_range = inexact_asarray(jnp.arange(1, PT - 1))
            sin_terms = jnp.sin(i_range * HT) * CLINC
            obj = jnp.sum(sin_terms[:, None] * x[1 : PT - 1, 1 : PY - 1])

        # Pre-compute weight array once
        cos_i = jnp.cos(inexact_asarray(jnp.arange(PT)) * HT)
        w_vals = (EX * cos_i + inexact_asarray(1.0)) ** 3

        if PT > 1 and PY > 1:
            # Lambda coefficients for right triangles - groups GR have scale 2.0
            lambda_c = (
                inexact_asarray(2.0) * w_vals[: PT - 1] + w_vals[1:PT]
            ) / inexact_asarray(6.0)

            # A(I,J) elements: (X(I+1,J) - X(I,J))^2 with weight LA/HT2
            # B(I,J) elements: (X(I,J+1) - X(I,J))^2 with weight LA/HY2
            # Apply 2.0 group scaling factor
            h_diffs = x[1:, : PY - 1] - x[: PT - 1, : PY - 1]  # A elements
            v_diffs = x[: PT - 1, 1:] - x[: PT - 1, : PY - 1]  # B elements

            obj += jnp.sum(
                lambda_c[:, None] * (HY_HT * h_diffs**2 + HT_HY * v_diffs**2)
            )

            # Mu coefficients for left triangles
            # Handle w_left array correctly for all indices
            if PT == 2:
                w_left = w_vals[0:1]  # Just the first element
            else:
                # For i=1 (index 0), use w_vals[0]. For i>=2, use w_vals[i-2]
                w_left = jnp.concatenate([w_vals[0:1], w_vals[: PT - 2]])
            mu_c = (inexact_asarray(2.0) * w_vals[1:PT] + w_left) / inexact_asarray(6.0)

            # C(I,J) elements: (X(I-1,J) - X(I,J))^2 with weight MU/HT2
            # D(I,J) elements: (X(I,J-1) - X(I,J))^2 with weight MU/HY2
            h_diffs_rev = x[: PT - 1, 1:] - x[1:PT, 1:]  # C elements
            v_diffs_rev = x[1:PT, : PY - 1] - x[1:PT, 1:PY]  # D elements

            obj += jnp.sum(
                mu_c[:, None] * (HY_HT * h_diffs_rev**2 + HT_HY * v_diffs_rev**2)
            )

        return obj

    @property
    def y0(self):
        """Initial guess."""
        PT = self.PT
        PY = self.PY

        # Constants matching objective function - ensure proper dtypes
        LT = inexact_asarray(8.0 * jnp.pi / 4.0)
        HT = LT / inexact_asarray(PT - 1)

        # Initialize boundary to zero, interior based on sin(xi)
        x0 = jnp.zeros((PT, PY), dtype=LT.dtype)

        if PT > 2 and PY > 2:
            i_vals = inexact_asarray(jnp.arange(1, PT - 1))
            sin_vals = jnp.sin(i_vals * HT)
            x0 = x0.at[1 : PT - 1, 1 : PY - 1].set(sin_vals[:, None])

        return x0.reshape(-1)

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Variable bounds - vectorized implementation."""
        PT, PY = self.PT, self.PY

        # All variables are positive (per SIF: "Other variables are positive")
        # but boundary variables are fixed at 0.0
        lower_2d = jnp.zeros((PT, PY))
        upper_2d = jnp.full((PT, PY), jnp.inf)

        # Set fixed boundary conditions:
        # Top and bottom edges (i=0 and i=PT-1) are fixed at 0.0
        upper_2d = upper_2d.at[0, :].set(0.0)  # X(1,J) = 0
        upper_2d = upper_2d.at[PT - 1, :].set(0.0)  # X(PT,J) = 0

        # Left and right edges (j=0 and j=PY-1) for interior rows are fixed at 0.0
        if PT > 2:
            upper_2d = upper_2d.at[1 : PT - 1, 0].set(0.0)  # X(I,1) = 0
            upper_2d = upper_2d.at[1 : PT - 1, PY - 1].set(0.0)  # X(I,PY) = 0

        return lower_2d.reshape(-1), upper_2d.reshape(-1)

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file comments:
        if self.PT == 4 and self.PY == 4:
            return inexact_asarray(-0.2247400)
        elif self.PT == 10 and self.PY == 10:
            return inexact_asarray(-0.1789600)
        elif self.PT == 23 and self.PY == 23:
            return inexact_asarray(-0.1800500)
        elif self.PT == 32 and self.PY == 32:
            return inexact_asarray(-0.1803000)
        elif self.PT == 75 and self.PY == 75:
            return inexact_asarray(-0.1805500)
        elif self.PT == 100 and self.PY == 100:
            return inexact_asarray(-0.1805700)
        elif self.PT == 125 and self.PY == 125:
            return inexact_asarray(-0.1805800)
        return None
