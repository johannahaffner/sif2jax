from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class HATFLDFLNE(AbstractNonlinearEquations):
    """Fletcher's variation of a test problem (HATFLDF) from the OPTIMA user manual.
    Monotonic paths to the solution from the initial point move to infinity
    and back

    Source:
    "The OPTIMA user manual (issue No.8, p. 47)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990, mods Nick Gould, August 2008
    Nonlinear-equations version of HATFLDFK.SIF, Nick Gould, Jan 2020.

    classification NOR2-AN-3-3
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "3"]:
        """Residual function for the nonlinear equations."""
        x1, x2, x3 = y

        # Constants
        c = jnp.array([0.032, 0.056, 0.099])

        residuals = []
        for i in range(1, 4):  # i = 1, 2, 3
            # G(i): X1 + X2 * X3^i = C(i)
            res = x1 + x2 * (x3**i) - c[i - 1]
            residuals.append(res)

        return jnp.array(residuals)

    def y0(self) -> Float[Array, "3"]:
        """Initial guess for the optimization problem."""
        # Fletcher's nasty starting point
        return jnp.array([1.2, -1.2, 0.98])

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "3"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
