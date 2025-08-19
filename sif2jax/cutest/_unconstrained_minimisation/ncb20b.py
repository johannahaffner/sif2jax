import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class NCB20B(AbstractUnconstrainedMinimisation):
    """A banded problem with semi-bandwidth 20.

    This problem exhibits frequent negative curvature in the exact Hessian.
    It is a simplified version of problem NCB20.

    Source:
    Ph. Toint, private communication, 1993.

    SIF input: Ph. Toint, April 1993.

    classification OUR2-AN-V-0

    TODO: Human review needed - objective differs by ~9000 at ones vector
    Attempts made: Checked all components (constants, banded linear, BP, QR)
    Suspected issues: Possible misinterpretation of SIF RD operator or GROUP USES
    """

    n: int = 5000  # Default problem size
    p: int = 20  # Semi-bandwidth
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Initial guess - all zeros."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Objective function with banded structure and nonlinear elements.

        Fully vectorized implementation.
        """
        del args

        n = self.n
        p = self.p
        x = y

        # Constants
        rp = float(p)
        cl = -4.0 / rp

        # Constants: Each group O(i) has constant -2.0, for i=1..n
        # Note: SIF constant seems to be negated in pycutest
        obj = 2.0 * n

        # Linear terms from banded structure - vectorized
        # O(i) += cl * sum(x[i+j-1] for j in 1..p) for i in 1..n-p+1
        n_minus_p_plus_1 = n - p + 1
        # Create matrix of indices for all band elements
        # Each row i contains indices [i, i+1, ..., i+p-1]
        band_indices = jnp.arange(n_minus_p_plus_1)[:, None] + jnp.arange(p)[None, :]
        # Sum all elements in the band
        obj += cl * jnp.sum(x[band_indices])

        # BP element contributions - fully vectorized
        # Create sliding windows of size p for all positions
        # Shape will be (n_minus_p_plus_1, p) containing all sliding windows

        # Create index matrix and gather
        window_indices = jnp.arange(n_minus_p_plus_1)[:, None] + jnp.arange(p)[None, :]
        windows = x[window_indices]  # Shape: (n_minus_p_plus_1, p)

        # Compute y_k = v_k / (1 + v_k^2) for all windows at once
        s = 1.0
        d_vals = s + windows * windows
        y_vals = windows / d_vals

        # Sum within each window
        sum_y = jnp.sum(y_vals, axis=1)  # Shape: (n_minus_p_plus_1,)

        # Weights: 1/(10*(i+1)) for i in 0..n_minus_p
        weights = 1.0 / (10.0 * jnp.arange(1, n_minus_p_plus_1 + 1, dtype=y.dtype))

        # Weighted sum of squares
        obj += jnp.sum(weights * sum_y * sum_y)

        # QR element contributions (x^4 terms) with coefficient 100.0
        obj += 100.0 * jnp.sum(x**4)

        return obj

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided."""
        return None
