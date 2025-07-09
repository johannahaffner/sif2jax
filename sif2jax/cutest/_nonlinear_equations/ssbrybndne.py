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

        # Compute scaling factors
        scale = jnp.zeros(n)
        for i in range(n):
            rat = float(i) / float(n - 1)
            arg = rat * scal
            scale = scale.at[i].set(jnp.exp(arg))

        # Initialize residuals
        residuals = jnp.zeros(n)

        # Compute residuals for each equation
        for i in range(n):
            # Linear part
            linear_part = 0.0

            # Determine the range of j indices
            j_start = max(0, i - lb)
            j_end = min(n - 1, i + ub)

            # Sum over j
            for j in range(j_start, j_end + 1):
                if j == i:
                    linear_part += kappa1 * scale[i] * x[i]
                else:
                    linear_part -= kappa3 * scale[j] * x[j]

            # Nonlinear part
            nonlinear_part = 0.0
            for j in range(n):
                if j != i:
                    xj_cubed = x[j] * x[j] * x[j]
                    nonlinear_part += kappa2 * scale[j] * xj_cubed

            # Full residual
            xi_plus_1 = 1.0 + x[i]
            xi_plus_1_cubed = xi_plus_1 * xi_plus_1 * xi_plus_1
            residuals = residuals.at[i].set(
                linear_part + nonlinear_part + kappa2 * scale[i] * xi_plus_1_cubed
            )

        return residuals

    def y0(self) -> Float[Array, "5000"]:
        """Initial guess for the optimization problem."""
        n = self.n
        return -jnp.ones(n)

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
