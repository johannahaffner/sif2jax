import jax
import jax.numpy as jnp
import jax.scipy.special as jss

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed
# Attempts made: [reviewed implementation for numerical stability]
# Suspected issues: [NaN values in gradients/hessians due to numerical instability in
# erfc/exp computations, possibly from very small error values causing large weights]
# Additional resources needed: [numerical analysis expert review, reference
# implementation comparison]
class CERI651ALS(AbstractUnconstrainedMinimisation):
    """ISIS Data fitting problem CERI651A given as an inconsistent set of
    nonlinear equations.

    Fit: y = c + l * x + I*A*B/2(A+B) *
         [ exp( A*[A*S^2+2(x-X0)]/2) * erfc( A*S^2+(x-X0)/S*sqrt(2) ) +
           exp( B*[B*S^2+2(x-X0)]/2) * erfc( B*S^2+(x-X0)/S*sqrt(2) ) ]

    Source: fit to a sum of a linear background and a back-to-back exponential
    using data enginx_ceria193749_spectrum_number_651_vana_corrected-0
    from Mantid (http://www.mantidproject.org)

    subset X in [36844.7449265, 37300.5256846]

    SIF input: Nick Gould and Tyrone Rees, Mar 2016
    Least-squares version of CERI651A.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-7-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 7  # Number of variables
    m: int = 61  # Number of data points

    def objective(self, y, args):
        del args
        c, l, a, b, i, s, x0 = y

        # X data points from the SIF file (lines 49-109)
        x_data = jnp.array(
            [
                36850.62500,
                36858.00000,
                36865.37500,
                36872.75000,
                36880.12500,
                36887.50000,
                36894.87500,
                36902.25000,
                36909.62500,
                36917.00000,
                36924.37500,
                36931.75000,
                36939.12500,
                36946.50000,
                36953.87500,
                36961.26563,
                36968.67188,
                36976.07813,
                36983.48438,
                36990.89063,
                36998.29688,
                37005.70313,
                37013.10938,
                37020.51563,
                37027.92188,
                37035.32813,
                37042.73438,
                37050.14063,
                37057.54688,
                37064.95313,
                37072.35938,
                37079.76563,
                37087.17188,
                37094.57813,
                37101.98438,
                37109.39063,
                37116.81250,
                37124.25000,
                37131.68750,
                37139.12500,
                37146.56250,
                37154.00000,
                37161.43750,
                37168.87500,
                37176.31250,
                37183.75000,
                37191.18750,
                37198.62500,
                37206.06250,
                37213.50000,
                37220.93750,
                37228.37500,
                37235.81250,
                37243.25000,
                37250.68750,
                37258.12500,
                37265.56250,
                37273.01563,
                37280.48438,
                37287.95313,
                37295.42188,
            ]
        )

        # Y data points from the SIF file (lines 111-171)
        y_data = jnp.array(
            [
                0.00000000,
                1.96083316,
                2.94124974,
                0.98041658,
                5.88249948,
                1.96083316,
                3.92166632,
                3.92166632,
                3.92166632,
                4.90208290,
                2.94124974,
                14.70624870,
                15.68666528,
                21.56916476,
                41.17749637,
                64.70749429,
                108.82624040,
                132.35623832,
                173.53373469,
                186.27915023,
                224.51539686,
                269.61455955,
                256.86914400,
                268.63414297,
                293.14455747,
                277.45789219,
                211.76998132,
                210.78956474,
                176.47498443,
                151.96456993,
                126.47373884,
                80.39415957,
                95.10040828,
                71.57041035,
                65.68791087,
                37.25583005,
                40.19707979,
                25.49083108,
                22.54958134,
                26.47124766,
                19.60833160,
                20.58874818,
                14.70624870,
                11.76499896,
                6.86291606,
                4.90208290,
                1.96083316,
                6.86291606,
                8.82374922,
                0.98041658,
                1.96083316,
                3.92166632,
                5.88249948,
                7.84333264,
                3.92166632,
                3.92166632,
                3.92166632,
                2.94124974,
                0.98041658,
                0.98041658,
                2.94124974,
            ]
        )

        # Error data points from the SIF file (lines 173-233)
        e_data = jnp.array(
            [
                1.00000000,
                1.41421356,
                1.73205081,
                1.00000000,
                2.44948974,
                1.41421356,
                2.00000000,
                2.00000000,
                2.00000000,
                2.23606798,
                1.73205081,
                3.87298335,
                4.00000000,
                4.69041576,
                6.48074070,
                8.12403840,
                0.53565375,
                1.61895004,
                3.30413470,
                3.78404875,
                5.13274595,
                6.58312395,
                6.18641406,
                6.55294536,
                7.29161647,
                6.82260384,
                4.69693846,
                4.66287830,
                3.41640786,
                2.44989960,
                1.35781669,
                9.05538514,
                9.84885780,
                8.54400375,
                8.18535277,
                6.16441400,
                6.40312424,
                5.09901951,
                4.79583152,
                5.19615242,
                4.47213595,
                4.58257569,
                3.87298335,
                3.46410162,
                2.64575131,
                2.23606798,
                1.41421356,
                2.64575131,
                3.00000000,
                1.00000000,
                1.41421356,
                2.00000000,
                2.44948974,
                2.82842712,
                2.00000000,
                2.00000000,
                2.00000000,
                1.73205081,
                1.00000000,
                1.00000000,
                1.73205081,
            ]
        )

        # Define helper function for the erfc_scaled function
        def erfc_scaled(z):
            # erfc_scaled(z) = exp(z^2) * erfc(z)
            # For numerical stability, use scipy's erfcx which is exp(x^2)*erfc(x)
            # JAX doesn't have erfcx, so we need to be careful with large z
            # For |z| > 5, erfc(z) is very small and exp(z^2) is very large
            # Use asymptotic expansion for large |z|
            abs_z = jnp.abs(z)

            # For small |z|, use the direct formula
            small_z_result = jnp.exp(z * z) * (1.0 - jss.erf(z))

            # For large |z|, use asymptotic approximation
            # erfcx(z) ≈ 1/(sqrt(pi)*z) for large |z|
            large_z_result = 1.0 / (jnp.sqrt(jnp.pi) * abs_z)

            # Use where to select based on magnitude
            return jnp.where(abs_z < 5.0, small_z_result, large_z_result)

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
            erfc_arg_a = (a * s_safe * s_safe + diff) / (s_safe * jnp.sqrt(2.0))
            erfc_arg_b = (b * s_safe * s_safe + diff) / (s_safe * jnp.sqrt(2.0))

            # Compute the terms
            term1 = jnp.exp(exp_arg_a) * (1.0 - jss.erf(erfc_arg_a))
            term2 = jnp.exp(exp_arg_b) * (1.0 - jss.erf(erfc_arg_b))

            # Full back-to-back exponential
            b2b = prefactor * (term1 + term2)

            return background + b2b

        # Compute model predictions for all x values using vmap
        y_pred = jax.vmap(compute_model)(x_data)

        # Compute weighted residuals and return sum of squares
        residuals = weights * (y_pred - y_data)
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (START1)
        return jnp.array([0.0, 0.0, 1.0, 0.05, 26061.4, 38.7105, 37027.1])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # The SIF file doesn't specify the optimal objective value
        return None
