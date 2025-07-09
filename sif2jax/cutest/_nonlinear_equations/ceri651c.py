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


class CERI651C(AbstractNonlinearEquations):
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

    classification NOR2-MN-7-56
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "56"]:
        """Residual function for the nonlinear equations."""
        c, l, a, b, i_param, s, x0 = y

        # Data values
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

        # Compute model values using back-to-back exponential
        rootp5 = jnp.sqrt(0.5)
        apb = a + b
        p = 0.5 * i_param * a * b / apb

        residuals = []
        for i in range(56):
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
        return jnp.array([0.0, 0.0, 1.0, 0.05, 597.076, 22.9096, 24027.5])

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
