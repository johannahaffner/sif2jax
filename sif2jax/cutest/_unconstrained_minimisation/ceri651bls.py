import jax
import jax.numpy as jnp
import jax.scipy.special as jss

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CERI651BLS(AbstractUnconstrainedMinimisation):
    """ISIS Data fitting problem CERI651B given as an inconsistent set of
    nonlinear equations.

    Fit: y = c + l * x + I*A*B/2(A+B) *
         [ exp( A*[A*S^2+2(x-X0)]/2) * erfc( A*S^2+(x-X0)/S*sqrt(2) ) +
           exp( B*[B*S^2+2(x-X0)]/2) * erfc( B*S^2+(x-X0)/S*sqrt(2) ) ]

    Source: fit to a sum of a linear background and a back-to-back exponential
    using data enginx_ceria193749_spectrum_number_651_vana_corrected-0
    from Mantid (http://www.mantidproject.org)

    subset X in [26047.3026604, 26393.719109]

    SIF input: Nick Gould and Tyrone Rees, Mar 2016
    Least-squares version of CERI651B.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-7-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 7  # Number of variables
    m: int = 66  # Number of data points

    def objective(self, y, args):
        del args
        c, l, a, b, i, s, x0 = y

        # X data points from the SIF file
        x_data = jnp.array(
            [
                26052.42188,
                26057.64063,
                26062.85938,
                26068.07813,
                26073.29688,
                26078.51563,
                26083.73438,
                26088.95313,
                26094.17188,
                26099.39063,
                26104.60938,
                26109.82813,
                26115.04688,
                26120.26563,
                26125.48438,
                26130.70313,
                26135.92188,
                26141.14063,
                26146.35938,
                26151.57813,
                26156.79688,
                26162.01563,
                26167.23438,
                26172.45313,
                26177.68750,
                26182.93750,
                26188.18750,
                26193.43750,
                26198.68750,
                26203.93750,
                26209.18750,
                26214.43750,
                26219.68750,
                26224.93750,
                26230.18750,
                26235.43750,
                26240.68750,
                26245.93750,
                26251.18750,
                26256.43750,
                26261.68750,
                26266.93750,
                26272.18750,
                26277.43750,
                26282.68750,
                26287.93750,
                26293.18750,
                26298.43750,
                26303.68750,
                26308.93750,
                26314.18750,
                26319.43750,
                26324.68750,
                26329.93750,
                26335.18750,
                26340.43750,
                26345.68750,
                26350.93750,
                26356.18750,
                26361.43750,
                26366.68750,
                26371.93750,
                26377.18750,
                26382.43750,
                26387.68750,
                26392.93750,
            ]
        )

        # Y data points from the SIF file
        y_data = jnp.array(
            [
                3.92166632,
                0.98041658,
                3.92166632,
                10.78458290,
                10.78458290,
                7.84333264,
                5.88249948,
                4.90208290,
                6.86291606,
                5.88249948,
                6.86291606,
                12.74541264,
                10.78458290,
                14.70624870,
                23.53000100,
                43.13833262,
                59.80500099,
                95.10040828,
                136.27790466,
                182.35748392,
                243.14289583,
                293.14455747,
                356.87121712,
                366.67538600,
                377.46080514,
                375.49997198,
                340.20455810,
                305.89039448,
                242.16247925,
                196.08289998,
                153.92539701,
                115.68915039,
                93.13956904,
                79.41373691,
                59.80500099,
                43.13833262,
                33.33416374,
                27.45166426,
                27.45166426,
                19.60833160,
                19.60833160,
                19.60833160,
                14.70624870,
                12.74541264,
                4.90208290,
                4.90208290,
                7.84333264,
                8.82374922,
                7.84333264,
                2.94124974,
                1.96083316,
                0.00000000,
                3.92166632,
                1.96083316,
                0.98041658,
                1.96083316,
                0.98041658,
                1.96083316,
                3.92166632,
                1.96083316,
                0.98041658,
                0.00000000,
                1.96083316,
                1.96083316,
                0.00000000,
                0.98041658,
            ]
        )

        # Error data points
        e_data = jnp.array(
            [
                2.00000000,
                1.00000000,
                2.00000000,
                3.31662479,
                3.31662479,
                2.82842712,
                2.44948974,
                2.23606798,
                2.64575131,
                2.44948974,
                2.64575131,
                3.60555128,
                3.31662479,
                3.87298335,
                4.89897949,
                6.63325406,
                7.79102166,
                9.84885780,
                1.17564839,
                3.56648984,
                5.68680301,
                7.19792938,
                9.32953269,
                9.48835280,
                9.60234826,
                9.57602225,
                9.12141420,
                8.63710459,
                6.12065331,
                3.70536146,
                2.45772777,
                7.70621754,
                9.70539501,
                9.02776669,
                7.79102166,
                6.63325406,
                5.82842712,
                5.28726358,
                5.28726358,
                4.47213595,
                4.47213595,
                4.47213595,
                3.87298335,
                3.60555128,
                2.23606798,
                2.23606798,
                2.82842712,
                3.00000000,
                2.82842712,
                1.73205081,
                1.41421356,
                0.00000000,
                2.00000000,
                1.41421356,
                1.00000000,
                1.41421356,
                1.00000000,
                1.41421356,
                2.00000000,
                1.41421356,
                1.00000000,
                0.00000000,
                1.41421356,
                1.41421356,
                0.00000000,
                1.00000000,
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
        # Initial values similar to CERI651ALS
        return jnp.array([0.0, 0.0, 1.0, 0.05, 26061.4, 38.7105, 26227.1])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # The SIF file doesn't specify the optimal objective value
        return None
