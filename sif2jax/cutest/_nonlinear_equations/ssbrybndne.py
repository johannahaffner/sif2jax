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

        # Compute nonlinear part for all equations at once
        x_cubed = x * x * x
        # Sum of all scale[j] * x[j]^3
        total_nonlinear = kappa2 * jnp.sum(scale * x_cubed)
        # Subtract the diagonal part (scale[i] * x[i]^3) for each i
        nonlinear_parts = total_nonlinear - kappa2 * scale * x_cubed

        # Compute linear part using vectorized operations
        def compute_linear_part(i):
            # Determine the range of j indices
            j_start = jnp.maximum(0, i - lb)
            j_end = jnp.minimum(n - 1, i + ub)

            # Create mask for valid j indices
            j_indices = jnp.arange(n)
            mask = (j_indices >= j_start) & (j_indices <= j_end)

            # Compute linear contributions
            linear_contrib = jnp.where(
                j_indices == i,
                kappa1 * scale[i] * x[i],  # diagonal term
                -kappa3 * scale * x,  # off-diagonal terms
            )

            # Apply mask and sum
            return jnp.sum(linear_contrib * mask.astype(linear_contrib.dtype))

        # Vectorize linear part computation over all i
        linear_parts = jax.vmap(compute_linear_part)(indices)

        # Compute final residuals
        xi_plus_1 = 1.0 + x
        xi_plus_1_cubed = xi_plus_1 * xi_plus_1 * xi_plus_1
        residuals = linear_parts + nonlinear_parts + kappa2 * scale * xi_plus_1_cubed

        return residuals

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

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "5000"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
