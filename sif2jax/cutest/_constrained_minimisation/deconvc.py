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

    # TODO: Human review needed
    # Attempts made: Fixed dimension mismatch, adjusted bounds, handled fixed variables
    # Suspected issues: Gradient/Hessian discrepancy - indexing issue with C variables
    # Additional resources needed: Clarification on fixed variables C(-LGSG:0)
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        # Variables are C(1:LGTR) and SG(1:LGSG)
        # C(-LGSG:0) are fixed to 0 and not included
        lgtr = 40
        lgsg = 11
        return lgtr + lgsg  # 40 + 11 = 51

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
        # C(1:LGTR) are the first 40 variables
        c_positive = y[:lgtr]  # 40 values
        # SG(1:LGSG) are the next 11 variables
        sg = y[lgtr:]  # 11 values

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
        # C(-LGSG:0) = 0 (fixed), C(1:LGTR) = c_positive
        # In array indexing: c[0:lgsg+1] = 0, c[lgsg+1:lgsg+1+lgtr] = c_positive
        c_full = jnp.concatenate([jnp.zeros(lgsg + 1), c_positive])

        # Compute residuals using proper indexing
        # R(K) = sum(SG(I) * C(K-I+1) for I=1 to LGSG) - TR(K)
        # where K=1..LGTR (1-indexed) maps to k=0..lgtr-1 (0-indexed)
        # and I=1..LGSG (1-indexed) maps to i=0..lgsg-1 (0-indexed)

        def compute_residual(k):
            # For SIF: sum over I=1 to LGSG of SG(I)*C(K-I+1)
            # In 0-indexed: sum over i=0 to lgsg-1 of sg[i]*C(k-i+1)
            # C(k-i+1) in SIF maps to c_full[k-i+1+lgsg] in our array

            # Use dynamic slicing to get the relevant part of c_full
            # We need C(k-i+1) for i=0 to lgsg-1
            # Which is C(k+1), C(k), ..., C(k-lgsg+2)
            # In c_full indexing: c_full[k+1+lgsg], c_full[k+lgsg], ...,
            # c_full[k-lgsg+2+lgsg] = c_full[k+lgsg+1], c_full[k+lgsg], ..., c_full[k+2]
            # So we need to slice from k+2 for lgsg elements and reverse
            start = k + 2
            # Use lax.dynamic_slice for gradient-friendly indexing
            c_slice = jax.lax.dynamic_slice(c_full, (start,), (lgsg,))
            # Reverse the slice to get the right order
            c_slice_rev = c_slice[::-1]

            # Compute dot product
            return jnp.dot(sg, c_slice_rev) - tr[k]

        # Vectorize over all k values
        residuals = jax.vmap(compute_residual)(jnp.arange(lgtr))

        # Sum of squares
        obj = jnp.sum(residuals * residuals)

        return obj

    def constraint(self, y):
        """Compute the energy constraint."""
        lgtr = 40

        # Extract SG variables (last 11 variables)
        sg = y[lgtr:]

        # Energy constraint: sum(SG(I)^2) = PIC
        pic = 12.35
        energy = jnp.sum(sg * sg) - pic

        return jnp.array([energy]), None

    def equality_constraints(self):
        """Energy constraint is an equality."""
        return jnp.ones(1, dtype=bool)

    def y0(self):
        """Initial guess."""
        lgtr = 40

        # Initial C values for C(1:40) (all zeros as given)
        c_init = jnp.zeros(lgtr)

        # Initial SG values
        sg_init = jnp.array(
            [1e-2, 2e-2, 0.4, 0.6, 0.8, 3.0, 0.8, 0.6, 0.44, 1e-2, 1e-2]
        )

        return inexact_asarray(jnp.concatenate([c_init, sg_init]))

    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    def bounds(self):
        """Variable bounds."""
        # From pycutest behavior, all variables have lower bound 0
        lower = jnp.zeros(self.n)
        upper = jnp.full(self.n, jnp.inf)

        return lower, upper

    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None
