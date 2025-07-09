from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class BIGGS6NE(AbstractNonlinearEquations):
    """Biggs EXP problem in 6 variables. This is a nonlinear equation version
    of problem BIGGS6.

    Source: Problem 21 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.
    Modification as a set of nonlinear equations: Nick Gould, Oct 2015.

    classification NOR2-AN-6-13
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    def residual(self, y, args) -> Float[Array, "13"]:
        """Residual function for the nonlinear equations."""
        x1, x2, x3, x4, x5, x6 = y

        residuals = []
        for i in range(1, 14):  # i from 1 to 13
            t = -0.1 * i

            # Compute Y(i) = exp(-0.1*i) - 5*exp(-i) + 3*exp(-0.4*i)
            y_i = jnp.exp(t) - 5.0 * jnp.exp(-i) + 3.0 * jnp.exp(4.0 * t)

            # Compute residual
            term1 = x3 * jnp.exp(t * x1)
            term2 = x4 * jnp.exp(t * x2)
            term3 = x6 * jnp.exp(t * x5)

            residuals.append(term1 - term2 + term3 - y_i)

        return jnp.array(residuals)

    def y0(self) -> Float[Array, "6"]:
        """Initial guess for the optimization problem."""
        if self.y0_iD == 0:
            # BIGGS6 start point
            return jnp.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        else:
            # OTHERX start point
            return jnp.array([1.0, 2.0, 1.0, 1.0, 4.0, 3.0])

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "6"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide a solution
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
