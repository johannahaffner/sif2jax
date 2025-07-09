from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class MISRA1D(AbstractNonlinearEquations):
    """NIST Data fitting problem MISRA1D given as an inconsistent set of
    nonlinear equations.

    Fit: y = b1*b2*x*((1+b2*x)**(-1))

    Source:  Problem from the NIST nonlinear regression test set
      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Misra, D., NIST (1978).
      Dental Research Monomolecular Adsorption Study.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    classification NOR2-MN-2-14
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    def residual(self, y, args) -> Float[Array, "14"]:
        """Residual function for the nonlinear equations."""
        b1, b2 = y

        # Data values
        x_data = jnp.array(
            [
                77.6,
                114.9,
                141.1,
                190.8,
                239.9,
                289.0,
                332.8,
                378.4,
                434.8,
                477.3,
                536.8,
                593.1,
                689.1,
                760.0,
            ]
        )

        y_data = jnp.array(
            [
                10.07,
                14.73,
                17.94,
                23.93,
                29.61,
                35.18,
                40.02,
                44.82,
                50.76,
                55.05,
                61.01,
                66.40,
                75.47,
                81.78,
            ]
        )

        # Model: y = b1*b2*x / (1 + b2*x)
        model = b1 * b2 * x_data / (1.0 + b2 * x_data)

        # Residuals
        return model - y_data

    def y0(self) -> Float[Array, "2"]:
        """Initial guess for the optimization problem."""
        if self.y0_iD == 0:
            # START1
            return jnp.array([500.0, 0.0001])
        else:
            # START2
            return jnp.array([450.0, 0.0003])

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "2"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
