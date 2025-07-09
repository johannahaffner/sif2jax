from jax import numpy as jnp
from jax.scipy.special import erfc
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


def erfc_scaled(z):
    """Scaled complementary error function: exp(z^2) * erfc(z)"""
    # For large positive z, use asymptotic expansion to avoid overflow
    # For negative z, use the relationship: erfc_scaled(z) = exp(z^2) * erfc(z)
    return jnp.where(
        z > 5.0,
        1.0 / (jnp.sqrt(jnp.pi) * z),  # Leading term of asymptotic expansion
        jnp.exp(z * z) * erfc(z),
    )


class CERI651A(AbstractNonlinearEquations):
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

    classification NOR2-MN-7-61
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "61"]:
        """Residual function for the nonlinear equations."""
        c, l, a, b, i_param, s, x0 = y

        # Data values
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
                10.53565375,
                11.61895004,
                13.30413470,
                13.78404875,
                15.13274595,
                16.58312395,
                16.18641406,
                16.55294536,
                17.29161647,
                16.82260384,
                14.69693846,
                14.66287830,
                13.41640786,
                12.44989960,
                11.35781669,
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

        # Compute model values using back-to-back exponential
        rootp5 = jnp.sqrt(0.5)
        apb = a + b
        p = 0.5 * i_param * a * b / apb

        residuals = []
        for i in range(61):
            x = x_data[i]
            xmy = x - x0
            z = xmy / s

            # R term
            r = jnp.exp(-0.5 * z * z)

            # AC and BC terms
            ac = rootp5 * (a * s + xmy / s)
            bc = rootp5 * (b * s + xmy / s)

            # QA and QB using scaled erfc
            qa = erfc_scaled(ac)
            qb = erfc_scaled(bc)

            # Model value
            model = c + l * x + p * r * (qa + qb)

            # Residual weighted by error
            residuals.append((model - y_data[i]) / e_data[i])

        return jnp.array(residuals)

    def y0(self) -> Float[Array, "7"]:
        """Initial guess for the optimization problem."""
        return jnp.array([0.0, 0.0, 1.0, 0.05, 26061.4, 38.7105, 37027.1])

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "7"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
