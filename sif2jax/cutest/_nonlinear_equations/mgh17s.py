from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class MGH17S(AbstractNonlinearEquations):
    """NIST Data fitting problem MGH17 given as an inconsistent set of
    nonlinear equations (scaled version).

    Fit: y = b1 + b2*exp[-x*0.01*b4] + b3*exp[-x*0.01*b5] + e

    Source:  Problem from the NIST nonlinear regression test set
      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Osborne, M. R. (1972).
      Some aspects of nonlinear least squares calculations.
      In Numerical Methods for Nonlinear Optimization, Lootsma (Ed).
      New York, NY:  Academic Press, pp. 171-189.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    classification NOR2-MN-5-33
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    def residual(self, y, args) -> Float[Array, "33"]:
        """Residual function for the nonlinear equations."""
        b1, b2, b3, b4, b5 = y

        # Data values from SIF file
        x = jnp.array(
            [
                0.0,
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
                100.0,
                110.0,
                120.0,
                130.0,
                140.0,
                150.0,
                160.0,
                170.0,
                180.0,
                190.0,
                200.0,
                210.0,
                220.0,
                230.0,
                240.0,
                250.0,
                260.0,
                270.0,
                280.0,
                290.0,
                300.0,
                310.0,
                320.0,
            ]
        )
        y_data = jnp.array(
            [
                0.844,
                0.908,
                0.932,
                0.936,
                0.925,
                0.908,
                0.881,
                0.850,
                0.818,
                0.784,
                0.751,
                0.718,
                0.685,
                0.658,
                0.628,
                0.603,
                0.580,
                0.558,
                0.538,
                0.522,
                0.506,
                0.490,
                0.478,
                0.467,
                0.457,
                0.448,
                0.438,
                0.431,
                0.424,
                0.420,
                0.414,
                0.411,
                0.406,
            ]
        )

        # Model: y = b1 + b2*exp[-x*0.01*b4] + b3*exp[-x*0.01*b5]
        model_y = b1 + b2 * jnp.exp(-x * 0.01 * b4) + b3 * jnp.exp(-x * 0.01 * b5)
        return model_y - y_data

    @property
    def y0(self) -> Float[Array, "5"]:
        """Initial guess for the optimization problem."""
        if self.y0_iD == 0:
            # START1
            return jnp.array([50.0, 150.0, -100.0, 100.0, 200.0])
        else:
            # START2
            return jnp.array([0.5, 1.5, -1.0, 1.0, 2.0])

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
