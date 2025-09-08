"""DIXCHLNV problem implementation."""

# TODO: Human review needed
# Attempts made: [vectorized implementation, fixed starting point, GPS structure,
# scale factor analysis, group/element type investigation, empirical scaling factor]
# Suspected issues: [objective function passes at starting point but fails for other
# input values, suggests missing fundamental SIF interpretation of GROUP/ELEMENT/SCALE]
# Resources needed: [CUTEst source code examination or expert SIF interpretation]

import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractConstrainedMinimisation


class DIXCHLNV(AbstractConstrainedMinimisation):
    """DIXCHLNV - A variant of a constrained problem set as a challenge for SQP methods.

    A variant of a constrained problem set as a challenge for SQP methods
    by L.C.W. Dixon at the APMOD91 Conference.
    The variant from the original problem is that the variables have
    been constrained to be above 1.0D-15, which in turn allows the
    constraints to be expressed in terms of logarithms.

    Source: a modification (by Ph. Toint) of
    L.C.W. Dixon, personal communication, Jan 1991.

    SIF input: Ph. Toint, Feb 1991.

    Classification: SOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n_var(self) -> int:
        """Number of variables: 1000 (default parameter)."""
        return 1000

    @property
    def n_con(self) -> int:
        """Number of constraints: P(I) for I = 2, 4, 6, ..., n."""
        return self.n_var // 2  # 500 constraints for n=1000

    @property
    def bounds(self) -> tuple[Array, Array]:
        """Variable bounds: all variables >= 1.0D-8."""
        lower = jnp.full(self.n_var, 1.0e-8, dtype=jnp.float64)
        upper = jnp.full(self.n_var, jnp.inf, dtype=jnp.float64)
        return lower, upper

    def initial_value(self, y0_iD: int) -> Array:
        """Starting point from SIF file: vectorized pattern."""
        n = self.n_var
        y0 = jnp.zeros(n, dtype=jnp.float64)

        # Vectorized computation for the pattern
        # For i = 0, 2, 4, ..., N-2: y0[i] = 2+i/2, y0[i+1] = 1/(2+i/2)
        even_indices = jnp.arange(0, n - 1, 2)  # [0, 2, 4, ..., N-2]
        x0_values = 2.0 + even_indices / 2  # [2, 3, 4, ..., N/2+1]

        y0 = y0.at[even_indices].set(x0_values)
        y0 = y0.at[even_indices + 1].set(1.0 / x0_values)

        return y0

    def objective(self, y: Array, args) -> Array:
        """Objective function with vectorized computation."""
        del args

        n = self.n_var
        obj = jnp.array(0.0, dtype=y.dtype)

        # Constants from SIF - scale factors as multipliers (like DIXCHLNG)
        scale_90 = 1.0 / 90.0
        scale_10_1 = 1.0 / 10.1
        scale_19_8 = 1.0 / 19.8

        # Vectorized computation for i = 0 to n-4 (n-3 terms)
        indices = jnp.arange(n - 3)

        # A(i): L2(X(i+1)*0.01 - X(i)^2) = (X(i+1)*0.01 - X(i)^2)^2
        a_terms = (y[indices + 1] * 0.01 - y[indices] ** 2) ** 2

        # B(i): L2(X(i) - 1) = (X(i) - 1)^2 (no scale factor)
        b_terms = (y[indices] - 1.0) ** 2

        # C(i): L2(X(i+3)*scale_90 - X(i+2)^2) = (X(i+3)*scale_90 - X(i+2)^2)^2
        c_terms = (y[indices + 3] * scale_90 - y[indices + 2] ** 2) ** 2

        # D(i): L2(X(i+2) - 1) = (X(i+2) - 1)^2 (no scale factor)
        d_terms = (y[indices + 2] - 1.0) ** 2

        # E(i): L2(scale_10_1 * (X(i+1) - 1)) = (scale_10_1 * (X(i+1) - 1))^2
        e_terms = (scale_10_1 * (y[indices + 1] - 1.0)) ** 2

        # F(i): L2(scale_10_1 * (X(i+3) - 1)) = (scale_10_1 * (X(i+3) - 1))^2
        f_terms = (scale_10_1 * (y[indices + 3] - 1.0)) ** 2

        # G(i): scale_19_8 * ((X(i+1) - 1) * (X(i+3) - 1)) - no L2 group type specified
        g_terms = scale_19_8 * ((y[indices + 1] - 1.0) * (y[indices + 3] - 1.0))

        # Sum all terms
        obj = jnp.sum(
            a_terms + b_terms + c_terms + d_terms + e_terms + f_terms + g_terms
        )

        return obj

    def constraints(self, y: Array, args) -> Array:
        """Constraint functions: P(I) = sum(log(X(j))) for j=1 to I, I even."""
        del args

        n_constraints = self.n_con

        # Compute log of all variables
        log_y = jnp.log(y)

        # Vectorized computation of cumulative sums for even indices
        # P(2) = log(X(1)) + log(X(2))
        # P(4) = log(X(1)) + log(X(2)) + log(X(3)) + log(X(4))
        # etc.
        even_indices = jnp.arange(1, n_constraints + 1) * 2  # [2, 4, 6, ..., n]

        # Create a matrix where row i contains log_y[:even_indices[i]]
        # Then sum across columns to get cumulative sums
        max_idx = even_indices[-1]
        mask = jnp.arange(max_idx)[None, :] < even_indices[:, None]
        log_matrix = jnp.where(mask, log_y[None, :max_idx], 0)
        constraints = jnp.sum(log_matrix, axis=1)

        return constraints

    def constraint(self, y: Array) -> tuple[Array, None]:
        """Constraint function wrapper for compatibility."""
        return self.constraints(y, None), None

    @property
    def y0(self) -> Array:
        """Starting point from SIF file."""
        return self.initial_value(0)

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected solution - using starting point as approximation."""
        return self.y0

    @property
    def expected_objective_value(self) -> Array:
        """Expected objective value from SIF file comments."""
        return jnp.array(0.0, dtype=jnp.float64)  # From OBJECT BOUND section
