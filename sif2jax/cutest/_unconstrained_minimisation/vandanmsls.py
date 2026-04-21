import jax.lax as lax
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class VANDANMSLS(AbstractUnconstrainedMinimisation):
    """ISIS Data fitting problem VANDANIUM given as a least squares problem
    (trial version with subset of data).

    Fit: y = BS(x,b) + e

    Source: fit to a cubic B-spline to data
    vanadium_pattern_enginx236516_bank1.txt from Mantid
    (http://www.mantidproject.org)
    obtained from a bank of detectors of ISIS's ENGIN-X

    SIF input: Nick Gould and Tyrone Rees, Dec 2015
    Least-squares version of VANDANIUMS.SIF, Nick Gould, Jan 2020.

    classification SUR2-MN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = (
        22  # Number of variables (KNOTS - 1 = 20 - 1 + 3 = 22, from indices -1 to N+1)
    )
    m: int = 10  # Number of data points

    def objective(self, y, args=None):
        """Compute the least squares objective function."""
        # Data points
        x_data = jnp.array(
            [
                0.245569,
                0.245927,
                0.246285,
                0.246642,
                0.247,
                0.247358,
                0.248074,
                0.248431,
                0.248789,
                0.249147,
            ]
        )
        y_data = jnp.array(
            [
                0.262172,
                1.73783,
                0.960973,
                0.0390275,
                2.57713,
                1.42287,
                2.0,
                1.22819,
                0.771811,
                4.0,
            ]
        )
        e_data = jnp.array(
            [
                0.512028,
                1.31827,
                0.980292,
                0.197554,
                1.60534,
                1.19284,
                1.41421,
                1.10824,
                0.878528,
                2.0,
            ]
        )

        # B-spline parameters
        h = 5.5 / 19.0  # knot spacing: (xu - xl) / (knots - 1)
        twoh = 2.0 * h
        hh = h * h

        # Cubic B-spline basis matrix: d[k, j] = x_data[j] - k*h (xl=0)
        # k ranges from -1 to 20 (22 basis functions)
        k_vals = jnp.arange(-1, 21, dtype=y.dtype)
        d = x_data[None, :] - k_vals[:, None] * h  # (22, 10)
        d2 = d * d
        d3 = d2 * d

        # Piecewise cubic B-spline via select_n
        v0 = jnp.zeros_like(d)  # |d| >= 2h
        v1 = (twoh + d3) / 6.0  # -2h < d <= -h
        v2 = (twoh - d3) / 6.0  # h <= d < 2h
        v3 = twoh * hh / 3.0 - 0.5 * (twoh + d) * d2  # -h < d <= 0
        v4 = twoh * hh / 3.0 - 0.5 * (twoh - d) * d2  # 0 < d < h

        # fmt: off
        idx = jnp.where(d <= -twoh, 0,
              jnp.where(d <= -h, 1,
              jnp.where(d >= twoh, 0,
              jnp.where(d >= h, 2,
              jnp.where(d <= 0.0, 3, 4)))))
        # fmt: on
        basis = lax.select_n(idx, v0, v1, v2, v3, v4)  # (22, 10)

        # Weighted sum of basis functions, then scaled residuals
        residuals = (y @ basis - y_data) / e_data
        return jnp.sum(residuals**2)

    @property
    def y0(self):
        """Starting point - all coefficients start at 0.0."""
        return jnp.zeros(self.n)

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def bounds(self):
        """No bounds for this problem."""
        return None

    @property
    def expected_result(self):
        """Expected solution (not available for this problem)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value (not available for this problem)."""
        return None
