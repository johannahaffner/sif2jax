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


class CERI651B(AbstractNonlinearEquations):
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

    classification NOR2-MN-7-66
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "66"]:
        """Residual function for the nonlinear equations."""
        c, l, a, b, i_param, s, x0 = y

        # Data values
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
                26335.20313,
                26340.48438,
                26345.76563,
                26351.04688,
                26356.32813,
                26361.60938,
                26366.89063,
                26372.17188,
                26377.45313,
                26382.73438,
                26388.01563,
                26393.29688,
            ]
        )

        y_data = jnp.array(
            [
                1.96083316,
                0.98041658,
                0.00000000,
                1.96083316,
                3.92166632,
                0.00000000,
                3.92166632,
                0.98041658,
                1.96083316,
                1.96083316,
                3.92166632,
                1.96083316,
                0.98041658,
                4.90208290,
                8.82374922,
                5.88249948,
                14.70624870,
                12.74541554,
                27.45166424,
                27.45166424,
                32.35374715,
                52.94249533,
                43.13832953,
                47.05999585,
                50.00124559,
                61.76624455,
                52.94249533,
                34.31458031,
                42.15791295,
                32.35374715,
                40.19707979,
                33.33416373,
                23.52999792,
                16.66708186,
                12.74541554,
                13.72583212,
                13.72583212,
                12.74541554,
                6.86291606,
                14.70624870,
                7.84333264,
                8.82374922,
                8.82374922,
                6.86291606,
                9.80416580,
                4.90208290,
                4.90208290,
                3.92166632,
                1.96083316,
                2.94124974,
                2.94124974,
                0.00000000,
                0.98041658,
                0.98041658,
                3.92166632,
                0.98041658,
                0.98041658,
                1.96083316,
                1.96083316,
                0.00000000,
                3.92166632,
                0.98041658,
                0.98041658,
                0.98041658,
                2.94124974,
                0.00000000,
            ]
        )

        e_data = jnp.array(
            [
                1.41421356,
                1.00000000,
                1.00000000,
                1.41421356,
                2.00000000,
                1.00000000,
                2.00000000,
                1.00000000,
                1.41421356,
                1.41421356,
                2.00000000,
                1.41421356,
                1.00000000,
                2.23606798,
                3.00000000,
                2.44948974,
                3.87298335,
                3.60555128,
                5.29150262,
                5.29150262,
                5.74456265,
                7.34846923,
                6.63324958,
                6.92820323,
                7.14142843,
                7.93725393,
                7.34846923,
                5.91607978,
                6.55743852,
                5.74456265,
                6.40312424,
                5.83095189,
                4.89897949,
                4.12310563,
                3.60555128,
                3.74165739,
                3.74165739,
                3.60555128,
                2.64575131,
                3.87298335,
                2.82842712,
                3.00000000,
                3.00000000,
                2.64575131,
                3.16227766,
                2.23606798,
                2.23606798,
                2.00000000,
                1.41421356,
                1.73205081,
                1.73205081,
                1.00000000,
                1.00000000,
                1.00000000,
                2.00000000,
                1.00000000,
                1.00000000,
                1.41421356,
                1.41421356,
                1.00000000,
                2.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.73205081,
                1.00000000,
            ]
        )

        # Compute model values using back-to-back exponential
        rootp5 = jnp.sqrt(0.5)
        apb = a + b
        p = 0.5 * i_param * a * b / apb

        residuals = []
        for i in range(66):
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
        return jnp.array([0.0, 0.0, 1.0, 0.05, 3527.31, 29.4219, 26185.9])

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
