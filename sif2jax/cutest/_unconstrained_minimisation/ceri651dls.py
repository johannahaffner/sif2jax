import jax
import jax.numpy as jnp
from jax.scipy.special import erfc

from ..._problem import AbstractUnconstrainedMinimisation


def erfc_scaled(z):
    """Scaled complementary error function: exp(z^2) * erfc(z)

    Uses more accurate asymptotic expansion for large z.
    """
    # For large positive z, use more terms in asymptotic expansion
    # erfcx(z) ≈ 1/(sqrt(pi)*z) * (1 - 0.5/z^2 + 0.75/z^4 - ...)
    sqrt_pi = jnp.sqrt(jnp.pi)

    # Use more terms for better accuracy
    def asymptotic(z):
        z2 = z * z
        return 1.0 / (sqrt_pi * z) * (1.0 - 0.5 / z2 + 0.75 / (z2 * z2))

    return jnp.where(
        z > 5.0,
        asymptotic(z),
        jnp.exp(z * z) * erfc(z),
    )


# TODO: Human review needed - numerical stability issues
class CERI651DLS(AbstractUnconstrainedMinimisation):
    """ISIS Data fitting problem CERI651D given as an inconsistent set of
    nonlinear equations.

    Fit: y = c + l * x + I*A*B/2(A+B) *
         [ exp( A*[A*S^2+2(x-X0)]/2) * erfc( A*S^2+(x-X0)/S*sqrt(2) ) +
           exp( B*[B*S^2+2(x-X0)]/2) * erfc( B*S^2+(x-X0)/S*sqrt(2) ) ]

    Source: fit to a sum of a linear background and a back-to-back exponential
    using data enginx_ceria193749_spectrum_number_651_vana_corrected-0
    from Mantid (http://www.mantidproject.org)

    subset X in [12986.356148, 13161.356148]

    SIF input: Nick Gould and Tyrone Rees, Mar 2016
    Least-squares version of CERI651D.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-7-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 7  # Number of variables
    m: int = 67  # Number of data points

    def objective(self, y, args):
        del args
        c, l, a, b, i, s, x0 = y

        # X data points from the SIF file
        x_data = jnp.array(
            [
                12987.48438,
                12990.07813,
                12992.67188,
                12995.26563,
                12997.85938,
                13000.45313,
                13003.04688,
                13005.64063,
                13008.23438,
                13010.82813,
                13013.42188,
                13016.01563,
                13018.60938,
                13021.20313,
                13023.79688,
                13026.39063,
                13028.98438,
                13031.57813,
                13034.17188,
                13036.76563,
                13039.35938,
                13041.95313,
                13044.54688,
                13047.14063,
                13049.75000,
                13052.37500,
                13055.00000,
                13057.62500,
                13060.25000,
                13062.87500,
                13065.50000,
                13068.12500,
                13070.75000,
                13073.37500,
                13076.00000,
                13078.62500,
                13081.25000,
                13083.87500,
                13086.50000,
                13089.12500,
                13091.75000,
                13094.37500,
                13097.00000,
                13099.62500,
                13102.25000,
                13104.87500,
                13107.50000,
                13110.12500,
                13112.75000,
                13115.37500,
                13118.00000,
                13120.62500,
                13123.25000,
                13125.87500,
                13128.50000,
                13131.12500,
                13133.75000,
                13136.37500,
                13139.00000,
                13141.62500,
                13144.25000,
                13146.87500,
                13149.50000,
                13152.12500,
                13154.75000,
                13157.37500,
                13160.00000,
            ]
        )

        # Y data points from the SIF file
        y_data = jnp.array(
            [
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                1.96083316,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.98041658,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                4.90208290,
                0.98041658,
                0.98041658,
                0.98041658,
                3.92166632,
                1.96083316,
                1.96083316,
                0.98041658,
                1.96083316,
                1.96083316,
                1.96083316,
                0.98041658,
                0.98041658,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.98041658,
                0.98041658,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                1.96083316,
                0.98041658,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
            ]
        )

        # Error data points from the SIF file
        e_data = jnp.array(
            [
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.41421356,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                2.23606798,
                1.00000000,
                1.00000000,
                1.00000000,
                2.00000000,
                1.41421356,
                1.41421356,
                1.00000000,
                1.41421356,
                1.41421356,
                1.41421356,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.41421356,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
            ]
        )

        # Weights for weighted least squares (1/error)
        weights = 1.0 / e_data

        # Define function to compute model value for a single x
        def compute_model(x):
            # Linear background term
            background = c + l * x

            # Difference term
            diff = x - x0

            # Common term in the back-to-back exponential
            # Add small epsilon to avoid division by zero
            ab_sum = jnp.maximum(a + b, 1e-10)
            prefactor = i * a * b / (2.0 * ab_sum)

            # Based on the formula:
            # exp( A*[A*S^2+2(x-X0)]/2) * erfc( (A*S^2+(x-X0))/(S*sqrt(2)) )

            # Add small epsilon to s to avoid division by zero
            s_safe = jnp.maximum(s, 1e-10)

            # Arguments for the exponential functions
            exp_arg_a = 0.5 * a * (a * s_safe * s_safe + 2.0 * diff)
            exp_arg_b = 0.5 * b * (b * s_safe * s_safe + 2.0 * diff)

            # Clip to prevent overflow
            exp_arg_a = jnp.clip(exp_arg_a, -700, 700)
            exp_arg_b = jnp.clip(exp_arg_b, -700, 700)

            # Arguments for the erfc functions
            erfc_arg_a = (a * s_safe + diff / s_safe) / jnp.sqrt(2.0)
            erfc_arg_b = (b * s_safe + diff / s_safe) / jnp.sqrt(2.0)

            # Compute the terms using the scaled erfc function
            # Note: exp(exp_arg) * erfc(erfc_arg) = exp(exp_arg - erfc_arg^2) * erfcx
            term1 = jnp.exp(exp_arg_a - erfc_arg_a * erfc_arg_a) * erfc_scaled(
                erfc_arg_a
            )
            term2 = jnp.exp(exp_arg_b - erfc_arg_b * erfc_arg_b) * erfc_scaled(
                erfc_arg_b
            )

            # Full back-to-back exponential
            b2b = prefactor * (term1 + term2)

            return background + b2b

        # Compute model predictions for all x values using vmap
        y_pred = jax.vmap(compute_model)(x_data)

        # Compute weighted residuals and return sum of squares
        residuals = weights * (y_pred - y_data)
        return jnp.sum(residuals**2)

    @property
    def y0(self):
        # Initial values from the SIF file (START4)
        return jnp.array([0.0, 0.0, 1.0, 0.05, 15.1595, 8.0, 13072.9])

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    @property
    def expected_objective_value(self):
        # The SIF file doesn't specify the optimal objective value
        return None
