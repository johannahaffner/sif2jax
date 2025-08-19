import jax.numpy as jnp
from jax import lax

from ..._problem import AbstractUnconstrainedMinimisation


class NCB20(AbstractUnconstrainedMinimisation):
    """A banded problem with semi-bandwidth 20.

    TODO: Human review needed
    Attempts made: Initial implementation
    Suspected issues: Test failures detected but specific issue not captured
    Resources needed: Debug objective function or starting point

    This problem exhibits frequent negative curvature in the exact Hessian.

    Source:
    Ph. Toint, private communication, 1992.

    SIF input: Ph. Toint, October 1992.

    classification OUR2-AN-V-0
    """

    n: int = 5010  # Total variables (5000 X variables + 10 Y variables)
    p: int = 20  # Semi-bandwidth
    ny: int = 10  # Number of Y variables
    nx: int = 5000  # Number of X variables
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def y0(self):
        """Initial guess."""
        # First nx variables are zeros, last ny variables (Y(I)) start at 1.0
        x0 = jnp.zeros(self.nx)
        y0_vals = jnp.ones(self.ny)
        return jnp.concatenate([x0, y0_vals])

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Objective function with banded structure and nonlinear elements.

        Fully vectorized implementation using lax.scan for memory efficiency.
        """
        del args

        nx = self.nx
        p = self.p
        ny = self.ny

        # Split variables
        x = y[:nx]
        y_vars = y[nx : nx + ny]

        # Constants
        cond = 1.0e4
        inv_cond = 1.0 / cond
        rp = float(p)
        cl = -4.0 / rp

        # Constants: Each group O(i) has constant -2.0 for i=1..nx (N=5000)
        # Plus group W also gets the default constant -2.0
        # Note: SIF constant seems to be negated in pycutest
        obj = 2.0 * (nx + 1)

        # Linear terms from banded structure using lax.scan for memory efficiency
        n_minus_p = nx - p

        # For banded linear terms: compute efficiently by counting how many
        # windows each element appears in. Windows go from I=1 to N-P
        # (SIF 1-based) which is i=0 to n_minus_p-1 (0-based)
        # Window i accesses x[i] to x[i+p-1], for i in 0..n_minus_p-1
        # Element x[j] appears in windows max(0, j-p+1) to min(j, n_minus_p-1)
        # So x[j] appears in min(j+1, p, n_minus_p-j) windows

        # Compute number of windows each element appears in
        indices = jnp.arange(nx, dtype=y.dtype)
        # Number of windows starting at or before this index
        windows_from_start = jnp.minimum(indices + 1, n_minus_p)
        # Number of windows that can include this index (limited by window size)
        windows_from_size = jnp.minimum(indices + 1, p)
        # Number of windows ending at or after this index
        windows_from_end = jnp.maximum(n_minus_p - indices + p - 1, 0)
        # Take minimum of all constraints
        num_windows = jnp.minimum(
            jnp.minimum(windows_from_start, windows_from_size), windows_from_end
        )

        # Linear contribution from banded structure
        obj += cl * jnp.sum(num_windows * x)

        # BP element contributions using lax.scan for memory efficiency
        def bp_scan_fn(carry, i):
            # Extract window of size p starting at position i
            window = lax.dynamic_slice(x, [i], [p])

            # Compute y_k = v_k / (1 + v_k^2)
            s = 1.0
            d_vals = s + window * window
            y_vals = window / d_vals

            # Sum within window
            sum_y = jnp.sum(y_vals)

            # Weight for this window: 1/(10*(i+1))
            weight = 1.0 / (10.0 * jnp.float64(i + 1))

            # Add weighted square to accumulator
            contribution = weight * sum_y * sum_y

            return carry + contribution, None

        # Run scan over all windows
        bp_contribution, _ = lax.scan(
            bp_scan_fn, 0.0, jnp.arange(n_minus_p, dtype=jnp.int32)
        )
        obj += bp_contribution

        # QR element contributions (x^4 terms)
        obj += jnp.sum(x**4)

        # 3P element contributions - vectorized
        # Only first ny elements participate
        # For the default problem, ny=10 < nx=5000, so all Y variables participate
        x_first = x[:ny]
        x_offset = x[ny : 2 * ny]
        y_subset = y_vars[:ny]
        terms = x_first * x_offset * y_subset + 2.0 * y_subset * y_subset
        obj += inv_cond * jnp.sum(terms)

        return obj

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value not provided."""
        return None
