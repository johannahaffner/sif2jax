from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class MGH09(AbstractNonlinearEquations):
    """NIST Data fitting problem MGH09 given as an inconsistent set of
    nonlinear equations.

    Fit: y = b1*(x**2+x*b2) / (x**2+x*b3+b4) + e

    Source:  Problem from the NIST nonlinear regression test set
      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Kowalik, J.S., and M. R. Osborne, (1978).
      Methods for Unconstrained Optimization Problems.
      New York, NY:  Elsevier North-Holland.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    classification NOR2-MN-4-11
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    def residual(self, y, args) -> Float[Array, "11"]:
        """Residual function for the nonlinear equations."""
        b1, b2, b3, b4 = y

        # Data values from SIF file
        x = jnp.array(
            [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625]
        )
        y_data = jnp.array(
            [
                0.1957,
                0.1947,
                0.1735,
                0.160,
                0.0844,
                0.0627,
                0.0456,
                0.0342,
                0.0323,
                0.0235,
                0.0246,
            ]
        )

        # Model: y = b1*(x**2+x*b2) / (x**2+x*b3+b4)
        x2 = x * x
        numerator = b1 * (x2 + x * b2)
        denominator = x2 + x * b3 + b4

        model_y = numerator / denominator
        return model_y - y_data

    def y0(self) -> Float[Array, "4"]:
        """Initial guess for the optimization problem."""
        if self.y0_iD == 0:
            # START1
            return jnp.array([25.0, 39.0, 41.5, 39.0])
        else:
            # START2
            return jnp.array([0.25, 0.39, 0.415, 0.39])

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "4"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
