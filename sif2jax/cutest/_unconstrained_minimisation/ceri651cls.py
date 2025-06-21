import jax
import jax.numpy as jnp
import jax.scipy.special as jss

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CERI651CLS(AbstractUnconstrainedMinimisation):
    """ISIS Data fitting problem CERI651C given as an inconsistent set of
    nonlinear equations.

    Fit: y = c + l * x + I*A*B/2(A+B) *
         [ exp( A*[A*S^2+2(x-X0)]/2) * erfc( A*S^2+(x-X0)/S*sqrt(2) ) +
           exp( B*[B*S^2+2(x-X0)]/2) * erfc( B*S^2+(x-X0)/S*sqrt(2) ) ]

    Source: fit to a sum of a linear background and a back-to-back exponential
    using data enginx_ceria193749_spectrum_number_651_vana_corrected-0
    from Mantid (http://www.mantidproject.org)

    subset X in [23919.5789114, 24189.3183142]

    SIF input: Nick Gould and Tyrone Rees, Mar 2016
    Least-squares version of CERI651C.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-7-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 7  # Number of variables
    m: int = 56  # Number of data points

    def objective(self, y, args):
        del args
        c, l, a, b, i, s, x0 = y

        # X data points from the SIF file
        x_data = jnp.array(
            [
                23920.10938,
                23924.89063,
                23929.67188,
                23934.45313,
                23939.23438,
                23944.01563,
                23948.79688,
                23953.57813,
                23958.35938,
                23963.14063,
                23967.92188,
                23972.70313,
                23977.48438,
                23982.26563,
                23987.06250,
                23991.87500,
                23996.68750,
                24001.50000,
                24006.31250,
                24011.12500,
                24015.93750,
                24020.75000,
                24025.56250,
                24030.37500,
                24035.18750,
                24040.00000,
                24044.81250,
                24049.62500,
                24054.43750,
                24059.25000,
                24064.06250,
                24068.87500,
                24073.68750,
                24078.50000,
                24083.31250,
                24088.12500,
                24092.93750,
                24097.75000,
                24102.56250,
                24107.37500,
                24112.18750,
                24117.00000,
                24121.81250,
                24126.62500,
                24131.43750,
                24136.25000,
                24141.06250,
                24145.89063,
                24150.73438,
                24155.57813,
                24160.42188,
                24165.26563,
                24170.10938,
                24174.95313,
                24179.79688,
                24184.64063,
            ]
        )

        # Y data points from the SIF file
        y_data = jnp.array(
            [
                0.00000000,
                0.98041658,
                1.96083316,
                0.00000000,
                0.98041658,
                0.00000000,
                0.00000000,
                3.92166632,
                0.98041658,
                0.00000000,
                0.98041658,
                2.94124974,
                1.96083316,
                0.98041658,
                2.94124974,
                8.82374922,
                5.88249948,
                6.86291606,
                8.82374922,
                11.76499896,
                12.74541554,
                6.86291606,
                8.82374922,
                12.74541554,
                13.72583212,
                8.82374922,
                12.74541554,
                19.60833160,
                4.90208290,
                2.94124974,
                1.96083316,
                3.92166632,
                3.92166632,
                5.88249948,
                2.94124974,
                4.90208290,
                6.86291606,
                2.94124974,
                1.96083316,
                0.00000000,
                1.96083316,
                2.94124974,
                1.96083316,
                1.96083316,
                1.96083316,
                3.92166632,
                0.00000000,
                0.00000000,
                3.92166632,
                2.94124974,
                1.96083316,
                0.00000000,
                1.96083316,
                0.00000000,
                0.98041658,
                0.98041658,
            ]
        )

        # Error data points
        e_data = jnp.array(
            [
                1.00000000,
                1.00000000,
                1.41421356,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                2.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.73205081,
                1.41421356,
                1.00000000,
                1.73205081,
                3.00000000,
                2.44948974,
                2.64575131,
                3.00000000,
                3.46410162,
                3.60555128,
                2.64575131,
                3.00000000,
                3.60555128,
                3.74165739,
                3.00000000,
                3.60555128,
                4.47213595,
                2.23606798,
                1.73205081,
                1.41421356,
                2.00000000,
                2.00000000,
                2.44948974,
                1.73205081,
                2.23606798,
                2.64575131,
                1.73205081,
                1.41421356,
                1.00000000,
                1.41421356,
                1.73205081,
                1.41421356,
                1.41421356,
                1.41421356,
                2.00000000,
                1.00000000,
                1.00000000,
                2.00000000,
                1.73205081,
                1.41421356,
                1.00000000,
                1.41421356,
                1.00000000,
                1.00000000,
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
        # Initial values from the SIF file (START3)
        return jnp.array([0.0, 0.0, 1.0, 0.05, 597.076, 22.9096, 24027.5])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # The SIF file doesn't specify the optimal objective value
        return None
