from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class ENGVAL2NE(AbstractNonlinearEquations):
    """The ENGVAL2NE problem.

    Source: problem 15 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.
    Nonlinear-equations version of ENGVAL2.SIF, Nick Gould, Jan 2020.

    classification NOR2-AN-3-5
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "5"]:
        """Residual function for the nonlinear equations."""
        x1, x2, x3 = y

        # G1: X1^2 + X2^2 + X3^2 = 1.0
        res1 = x1**2 + x2**2 + x3**2 - 1.0

        # G2: X1^2 + X2^2 + (X3-2)^2 = 1.0
        res2 = x1**2 + x2**2 + (x3 - 2.0) ** 2 - 1.0

        # G3: X1 + X2 + X3 = 1.0
        res3 = x1 + x2 + x3 - 1.0

        # G4: X1 + X2 - X3 = -1.0
        res4 = x1 + x2 - x3 + 1.0

        # G5: 3*X2^2 + X1^3 + (5*X3 - X1 + 1)^2 = 36.0
        twow = 5.0 * x3 - x1 + 1.0
        res5 = 3.0 * x2**2 + x1**3 + twow**2 - 36.0

        return jnp.array([res1, res2, res3, res4, res5])

    def y0(self) -> Float[Array, "3"]:
        """Initial guess for the optimization problem."""
        return jnp.array([1.0, 2.0, 0.0])

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
