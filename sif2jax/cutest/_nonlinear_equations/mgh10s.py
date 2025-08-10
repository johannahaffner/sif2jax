from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class MGH10S(AbstractNonlinearEquations):
    """NIST Data fitting problem MGH10 given as an inconsistent set of
    nonlinear equations (scaled version).

    Fit: y = 0.01 * b1 * exp[ 1000 * b2 / (x + 100 * b3 ) ] + e

    Source:  Problem from the NIST nonlinear regression test set
      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Meyer, R. R. (1970).
      Theoretical and computational aspects of nonlinear
      regression.  In Nonlinear Programming, Rosen,
      Mangasarian and Ritter (Eds).
      New York, NY: Academic Press, pp. 465-486.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015
               correction by S. Gratton & Ph. Toint, May 2024

    classification NOR2-MN-3-16
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    def residual(self, y, args) -> Float[Array, "16"]:
        """Residual function for the nonlinear equations."""
        b1, b2, b3 = y

        # Data values from SIF file
        x = jnp.array(
            [
                50.0,
                55.0,
                60.0,
                65.0,
                70.0,
                75.0,
                80.0,
                85.0,
                90.0,
                95.0,
                100.0,
                105.0,
                110.0,
                115.0,
                120.0,
                125.0,
            ]
        )
        y_data = jnp.array(
            [
                34780.0,
                28610.0,
                23650.0,
                19630.0,
                16370.0,
                13720.0,
                11540.0,
                9744.0,
                8261.0,
                7030.0,
                6005.0,
                5147.0,
                4427.0,
                3820.0,
                3307.0,
                2872.0,
            ]
        )

        # Model: y = 0.01 * b1 * exp[ 1000 * b2 / (x + 100 * b3 ) ]
        model_y = 0.01 * b1 * jnp.exp(1000.0 * b2 / (x + 100.0 * b3))
        return model_y - y_data

    @property
    def y0(self) -> Float[Array, "3"]:
        """Initial guess for the optimization problem."""
        if self.y0_iD == 0:
            # START1
            return jnp.array([200.0, 400.0, 250.0])
        else:
            # START2
            return jnp.array([2.0, 4.0, 2.5])

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
