import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class DECONVC(AbstractConstrainedMinimisation):
    """DECONVC problem - deconvolution analysis (constrained version).

    A problem arising in deconvolution analysis.

    Source: J.P. Rasson, Private communication, 1996.

    SIF input: Ph. Toint, Nov 1996.

    Classification: SQR2-MN-61-1

    # Fixed: The issue was with the PROD element's IDX parameter handling.
    # The SIF file sets SCAL=0 when IDX<=0, which zeros out contributions
    # when K-I+1 <= 0. This is now properly implemented.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        # Variables are C(-LGSG:LGTR) and SG(1:LGSG)
        # C(-LGSG:0) are fixed to 0 but included
        lgtr = 40
        lgsg = 11
        return lgtr + lgsg + 12  # 40 + 11 + 12 = 63

    @property
    def m(self):
        """Number of constraints."""
        return 1  # Energy constraint

    def objective(self, y, args):
        """Compute the sum of squares objective."""
        del args

        lgtr = 40
        lgsg = 11

        # Extract variables
        # C(-LGSG:0) are the first 12 variables (fixed at 0)
        # C(1:LGTR) are the next 40 variables
        c_fixed = y[:12]  # 12 values (C(-11:0))
        c_positive = y[12 : 12 + lgtr]  # 40 values (C(1:40))
        # SG(1:LGSG) are the last 11 variables
        sg = y[12 + lgtr :]  # 11 values

        # Data values TR
        tr = jnp.array(
            [
                0.0,
                0.0,
                1.6e-3,
                5.4e-3,
                7.02e-2,
                0.1876,
                0.332,
                0.764,
                0.932,
                0.812,
                0.3464,
                0.2064,
                8.3e-2,
                3.4e-2,
                6.18e-2,
                1.2,
                1.8,
                2.4,
                9.0,
                2.4,
                1.801,
                1.325,
                7.62e-2,
                0.2104,
                0.268,
                0.552,
                0.996,
                0.36,
                0.24,
                0.151,
                2.48e-2,
                0.2432,
                0.3602,
                0.48,
                1.8,
                0.48,
                0.36,
                0.264,
                6e-3,
                6e-3,
            ]
        )

        # Create full C array
        # C(-LGSG:0) = c_fixed, C(1:LGTR) = c_positive
        # In array indexing: c[0:lgsg+1] = c_fixed, c[lgsg+1:lgsg+1+lgtr] = c_positive
        c_full = jnp.concatenate([c_fixed, c_positive])

        # Compute residuals using proper indexing
        # R(K) = sum(SG(I) * C(K-I+1) for I=1 to LGSG) - TR(K)
        # where K=1..LGTR (1-indexed) maps to k=0..lgtr-1 (0-indexed)
        # and I=1..LGSG (1-indexed) maps to i=0..lgsg-1 (0-indexed)

        def compute_residual(k):
            # For SIF: sum over I=1 to LGSG of SG(I)*C(K-I+1)
            # In 0-indexed: sum over i=0 to lgsg-1 of sg[i]*C(k-i+1)
            # IMPORTANT: The SIF file uses a special PR element that sets SCAL=0
            # when IDX<=0
            # This means when K-I+1 <= 0, the contribution is zero

            # For k (0-indexed), I goes from 1 to LGSG
            # So k-i+1 in 1-indexed goes from k+1 down to k-lgsg+2
            # We need to check which terms are positive

            # Create indices for i=0 to lgsg-1
            i_vals = jnp.arange(lgsg)
            # Calculate k-i+1 in 1-indexed (k is 0-indexed, so k+1 is 1-indexed K)
            indices_1based = (k + 1) - (i_vals + 1) + 1  # = k - i + 1

            # Only include terms where indices_1based > 0
            mask = indices_1based > 0

            # Map 1-based indices to 0-based c_full indices
            # C(-11) to C(0) are at indices 0 to 11
            # C(1) to C(40) are at indices 12 to 51
            # So C(idx) in 1-based maps to c_full[idx + lgsg] for idx > 0
            # and c_full[idx + lgsg] for idx <= 0 (since C(-11) is at index 0)
            c_indices = indices_1based + lgsg

            # Gather C values
            c_values = c_full[c_indices]

            # Apply mask (zero out negative index contributions)
            c_values_masked = jnp.where(mask, c_values, 0.0)

            # Compute dot product
            return jnp.dot(sg, c_values_masked) - tr[k]

        # Vectorize over all k values
        residuals = jax.vmap(compute_residual)(jnp.arange(lgtr))

        # Sum of squares
        obj = jnp.sum(residuals * residuals)

        return obj

    def constraint(self, y):
        """Compute the energy constraint."""
        # Extract SG variables (last 11 variables)
        sg = y[52:]  # After 12 fixed C values and 40 free C values

        # Energy constraint: sum(SG(I)^2) = PIC
        pic = 12.35
        energy = jnp.sum(sg * sg) - pic

        return jnp.array([energy]), None

    def equality_constraints(self):
        """Energy constraint is an equality."""
        return jnp.ones(1, dtype=bool)

    @property
    def y0(self):
        """Initial guess."""
        lgtr = 40

        # Initial C values for C(-11:0) (all zeros, fixed)
        c_fixed_init = jnp.zeros(12)
        # Initial C values for C(1:40) (all zeros as given)
        c_init = jnp.zeros(lgtr)

        # Initial SG values
        sg_init = jnp.array(
            [1e-2, 2e-2, 0.4, 0.6, 0.8, 3.0, 0.8, 0.6, 0.44, 1e-2, 1e-2]
        )

        return inexact_asarray(jnp.concatenate([c_fixed_init, c_init, sg_init]))

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """Variable bounds."""
        # From pycutest behavior, all variables have lower bound 0
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, jnp.inf)

        # C(-11:0) are fixed at 0.0 (first 12 variables)
        upper = upper.at[:12].set(0.0)

        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None
