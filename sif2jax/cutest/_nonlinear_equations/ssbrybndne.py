import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class SSBRYBNDNE(AbstractNonlinearEquations):
    """Broyden banded system of nonlinear equations, considered in the
    least square sense.
    NB: scaled version of BRYBND with scaling proposed by Luksan et al.
    This is a nonlinear equation variant of SSBRYBND

    Source: problem 48 in
    L. Luksan, C. Matonoha and J. Vlcek
    Modified CUTE problems for sparse unconstraoined optimization
    Technical Report 1081
    Institute of Computer Science
    Academy of Science of the Czech Republic

    that is a scaled variant of problem 31 in

    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#73 (p. 41) and Toint#18

    SIF input: Ph. Toint and Nick Gould, Nov 1997.
               Nick Gould (nonlinear equation version), Jan 2019

    classification NOR2-AN-V-V

    TODO: Human review needed
    Attempts made: Complex element structure with different patterns for corners
    Suspected issues: SQ/CB element usage pattern differs by region
    Resources needed: Verify GROUP USES pattern interpretation
    """

    n: int = 5000  # Default to n=5000
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "5000"]:
        """Residual function for the nonlinear equations."""
        x = y
        n = self.n

        # Problem parameters
        kappa1 = 2.0
        kappa2 = 5.0
        kappa3 = 1.0
        lb = 5
        ub = 1
        scal = 6.0

        # Compute scaling factors vectorized
        indices = jnp.arange(n)
        rat = indices / (n - 1)
        arg = rat * scal
        scale = jnp.exp(arg)

        # Element calculations
        # SQ elements (E): P^2 * V^2 where P = scale[j], V = x[j]
        scale_squared = scale * scale
        x_squared = x * x
        sq_elements = scale_squared * x_squared

        # CB elements (Q): P^3 * V^3 where P = scale[i], V = x[i]
        scale_cubed = scale * scale * scale
        x_cubed = x * x * x
        cb_elements = scale_cubed * x_cubed

        # Compute residuals for each equation using scan
        def compute_residual_i(carry, i):
            # Determine the range of j indices
            j_start = jnp.maximum(0, i - lb)
            j_end = jnp.minimum(n - 1, i + ub)

            # Create mask for j indices
            j_indices = jnp.arange(n)
            mask = (j_indices >= j_start) & (j_indices <= j_end) & (j_indices != i)

            # Linear part: kappa1 * scale[i] * x[i] for diagonal + off-diagonal terms
            res = kappa1 * scale[i] * x[i] + jnp.sum(-kappa3 * scale * x * mask)

            # Nonlinear part - different patterns for different regions
            # Upper left corner (i < lb)
            upper_left_nonlinear = kappa2 * cb_elements[i] + jnp.sum(
                -kappa3 * sq_elements * mask
            )

            # Middle part (lb <= i < n-ub)
            mask_less = mask & (j_indices < i)
            mask_greater = mask & (j_indices > i)
            middle_nonlinear = (
                kappa2 * sq_elements[i]
                + jnp.sum(-kappa3 * cb_elements * mask_less)
                + jnp.sum(-kappa3 * sq_elements * mask_greater)
            )

            # Lower right corner (i >= n-ub)
            lower_right_nonlinear = kappa2 * cb_elements[i] + jnp.sum(
                -kappa3 * sq_elements * mask
            )

            # Select appropriate nonlinear part based on region
            nonlinear_part = jnp.where(
                i < lb,
                upper_left_nonlinear,
                jnp.where(i < n - ub, middle_nonlinear, lower_right_nonlinear),
            )

            total_res = res + nonlinear_part
            return None, total_res

        _, residuals = jax.lax.scan(compute_residual_i, None, indices)

        return residuals

    @property
    def y0(self) -> Float[Array, "5000"]:
        """Initial guess for the optimization problem."""
        n = self.n
        scal = 6.0

        # Compute starting values: x[i] = 1 / scale[i] - vectorized
        indices = jnp.arange(n)
        rat = indices / (n - 1)
        arg = rat * scal
        scale = jnp.exp(arg)
        x0 = 1.0 / scale

        return x0

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
