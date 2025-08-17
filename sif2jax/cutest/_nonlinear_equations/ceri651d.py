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


class CERI651D(AbstractNonlinearEquations):
    """ISIS Data fitting problem CERI651D given as an inconsistent set of
    nonlinear equations.

    TODO: Human review needed
    Current status:
    - Starting values match SIF exactly ✓
    - Constraint dimensions correct ✓
    - Constraint values fail: numerical precision differences vs Fortran
    - Same issues as other CERI problems with exp/erfc computations

    Fit: y = c + l * x + I*A*B/2(A+B) *
               [ exp( A*[A*S^2+2(x-X0)]/2) * erfc( A*S^2+(x-X0)/S*sqrt(2) ) +
                 exp( B*[B*S^2+2(x-X0)]/2) * erfc( B*S^2+(x-X0)/S*sqrt(2) ) ]

    Source: fit to a sum of a linear background and a back-to-back exponential
    using data enginx_ceria193749_spectrum_number_651_vana_corrected-0
    from Mantid (http://www.mantidproject.org)

    subset X in [12986.356148, 13161.356148]

    SIF input: Nick Gould and Tyrone Rees, Mar 2016

    classification NOR2-MN-7-67
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "67"]:
        """Residual function for the nonlinear equations."""
        c, l, a, b, i_param, s, x0 = y

        # Data values
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
        # From START4 point in SIF file
        return jnp.array([0.0, 0.0, 1.0, 0.05, 15.1595, 8.0, 13072.9])

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
