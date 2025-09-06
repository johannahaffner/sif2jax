from jax import numpy as jnp
from jax.scipy.special import erfc
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


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


class CERI651E(AbstractNonlinearEquations):
    """ISIS Data fitting problem CERI651E given as an inconsistent set of
    nonlinear equations.

    TODO: Human review needed
    Current status:
    - Starting values match SIF exactly ✓
    - Constraint dimensions correct ✓
    - Constraint values fail: max difference ~0.0006 at starting point
    - Same issues as other CERI problems with exp/erfc computations

    Fit: y = c + l * x + I*A*B/2(A+B) *
               [ exp( A*[A*S^2+2(x-X0)]/2) * erfc( A*S^2+(x-X0)/S*sqrt(2) ) +
                 exp( B*[B*S^2+2(x-X0)]/2) * erfc( B*S^2+(x-X0)/S*sqrt(2) ) ]

    Source: fit to a sum of a linear background and a back-to-back exponential
    using data enginx_ceria193749_spectrum_number_651_vana_corrected-0
    from Mantid (http://www.mantidproject.org)

    subset X in [13556.2988352, 13731.2988352]

    SIF input: Nick Gould and Tyrone Rees, Mar 2016

    classification NOR2-MN-7-64
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "64"]:
        """Residual function for the nonlinear equations."""
        c, l, a, b, i_param, s, x0 = y

        # Data values
        x_data = jnp.array(
            [
                13558.04688,
                13560.76563,
                13563.48438,
                13566.20313,
                13568.92188,
                13571.64063,
                13574.35938,
                13577.07813,
                13579.79688,
                13582.51563,
                13585.23438,
                13587.95313,
                13590.67188,
                13593.39063,
                13596.10938,
                13598.82813,
                13601.54688,
                13604.26563,
                13606.98438,
                13609.70313,
                13612.42188,
                13615.14063,
                13617.85938,
                13620.57813,
                13623.29688,
                13626.01563,
                13628.73438,
                13631.45313,
                13634.17188,
                13636.89063,
                13639.60938,
                13642.32813,
                13645.04688,
                13647.76563,
                13650.48438,
                13653.20313,
                13655.92188,
                13658.64063,
                13661.35938,
                13664.07813,
                13666.79688,
                13669.51563,
                13672.23438,
                13674.96875,
                13677.71875,
                13680.46875,
                13683.21875,
                13685.96875,
                13688.71875,
                13691.46875,
                13694.21875,
                13696.96875,
                13699.71875,
                13702.46875,
                13705.21875,
                13707.96875,
                13710.71875,
                13713.46875,
                13716.21875,
                13718.96875,
                13721.71875,
                13724.46875,
                13727.21875,
                13729.96875,
            ]
        )

        y_data = jnp.array(
            [
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
                0.98041658,
                0.98041658,
                1.96083316,
                1.96083316,
                4.90208290,
                0.98041658,
                1.96083316,
                0.00000000,
                1.96083316,
                0.98041658,
                5.88249948,
                0.98041658,
                1.96083316,
                0.00000000,
                0.98041658,
                0.00000000,
                0.00000000,
                0.98041658,
                0.00000000,
                1.96083316,
                0.98041658,
                0.00000000,
                0.98041658,
                0.98041658,
                0.98041658,
                0.00000000,
                0.00000000,
                0.98041658,
                0.00000000,
                0.00000000,
                0.98041658,
                0.00000000,
                0.00000000,
                0.00000000,
                0.98041658,
                0.98041658,
                0.98041658,
                0.00000000,
                0.98041658,
                0.00000000,
                1.96083316,
                0.00000000,
                0.00000000,
            ]
        )

        e_data = jnp.array(
            [
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
                1.41421356,
                1.41421356,
                2.23606798,
                1.00000000,
                1.41421356,
                1.00000000,
                1.41421356,
                1.00000000,
                2.44948974,
                1.00000000,
                1.41421356,
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
                1.41421356,
                1.00000000,
                1.00000000,
            ]
        )

        # Compute model values using back-to-back exponential
        # From docstring:
        # y = c + l * x + I*A*B/2(A+B) *
        #     [ exp( A*[A*S^2+2(x-X0)]/2) * erfc( (A*S^2+(x-X0))/(S*sqrt(2)) ) +
        #       exp( B*[B*S^2+2(x-X0)]/2) * erfc( (B*S^2+(x-X0))/(S*sqrt(2)) ) ]

        sqrt2 = jnp.sqrt(2.0)
        apb = a + b
        p = 0.5 * i_param * a * b / apb

        # Vectorized computation
        xmy = x_data - x0

        # Terms for A exponential
        # Use scaled erfc to avoid overflow
        a_exp_arg = 0.5 * a * (a * s * s + 2 * xmy)
        a_erfc_arg = (a * s + xmy / s) / sqrt2
        # exp(a_exp_arg) * erfc(a_erfc_arg) =
        #   exp(a_exp_arg - a_erfc_arg^2) * exp(a_erfc_arg^2) * erfc(a_erfc_arg)
        qa = jnp.exp(a_exp_arg - a_erfc_arg * a_erfc_arg) * erfc_scaled(a_erfc_arg)

        # Terms for B exponential
        b_exp_arg = 0.5 * b * (b * s * s + 2 * xmy)
        b_erfc_arg = (b * s + xmy / s) / sqrt2
        qb = jnp.exp(b_exp_arg - b_erfc_arg * b_erfc_arg) * erfc_scaled(b_erfc_arg)

        # Model values
        model = c + l * x_data + p * (qa + qb)

        # Residuals weighted by error
        residuals = (model - y_data) / e_data

        return residuals

    @property
    def y0(self) -> Float[Array, "7"]:
        """Initial guess for the optimization problem."""
        # From START5 point in SIF file
        return jnp.array([0.0, 0.0, 1.0, 0.05, 17.06794, 8.0, 13642.3])

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self) -> None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    @property
    def expected_objective_value(self) -> Float[Array, ""]:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """No bounds for this problem."""
        return None
