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

    def _cubic_bspline(self, x0, k, h, x):
        """Compute the value of the k-th B-Spline at x.

        This implements the cubic B-spline function from the SIF file.
        """
        xx = x - k * h
        twoh = h + h  # 2*h

        def case1():  # xx <= x0 - 2*h or xx >= x0 + 2*h
            return jnp.array(0.0)

        def case2():  # xx <= x0 - h
            return (twoh + (xx - x0) ** 3) / 6.0

        def case3():  # xx >= x0 + h
            return (twoh - (xx - x0) ** 3) / 6.0

        def case4():  # xx <= x0
            return twoh * h * h / 3.0 - 0.5 * (twoh + xx - x0) * (xx - x0) ** 2

        def case5():  # xx > x0
            return twoh * h * h / 3.0 - 0.5 * (twoh - xx + x0) * (xx - x0) ** 2

        # Use jnp.where for conditional logic
        result = jnp.where(
            (xx <= x0 - twoh) | (xx >= x0 + twoh),
            case1(),
            jnp.where(
                xx <= x0 - h,
                case2(),
                jnp.where(xx >= x0 + h, case3(), jnp.where(xx <= x0, case4(), case5())),
            ),
        )

        return result

    def objective(self, y, args=None):
        """Compute the least squares objective function."""
        # Variables: A(-1), A(0), A(1), ..., A(20), A(21)
        # Total: 23 variables indexed from -1 to 21
        # But n=22 according to SIF, so we have 22 variables: A(-1) to A(20)
        a_coeffs = y  # Shape: (22,)

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

        # B-spline parameters from SIF file
        xl = 0.0  # Lower knot
        xu = 5.5  # Upper knot
        knots = 20  # Number of knots
        n_vars = knots - 1  # n = 19 (but we have 22 variables from -1 to 20)
        h = (xu - xl) / n_vars  # Knot spacing

        # Compute B-spline approximation for each data point
        def compute_bspline_value(x_val):
            # Sum over all B-spline basis functions
            # K ranges from -1 to N+1, where N = KNOTS - 1 = 19
            # So K ranges from -1 to 20, giving us 22 terms
            total = 0.0
            for k_idx in range(22):  # Indices 0 to 21 in our array
                k = k_idx - 1  # Convert to actual k values: -1 to 20
                a_k = a_coeffs[k_idx]
                basis_val = self._cubic_bspline(xl, k, h, x_val)
                total += a_k * basis_val
            return total

        # Vectorize the computation over all data points
        bspline_values = jnp.array([compute_bspline_value(x) for x in x_data])

        # Compute scaled residuals: (model - observed) / e_data
        residuals = (bspline_values - y_data) / e_data

        # Return sum of squared residuals for least squares
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
