from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class SINVALNE(AbstractNonlinearEquations):
    """A trigonometric variant of the 2 variables Rosenbrock
    "banana valley" problem.  This problem is a nonlinear
    equation version of problem SINEVAL.

    Source:  problem 4.2 in
    Y. Xiao and F. Zhou,
    "Non-monotone trust region methods with curvilinear path
    in unconstrained optimization",
    Computing, vol. 48, pp. 303-317, 1992.

    SIF input: F Facchinei, M. Roma and Ph. Toint, June 1994

    classification NOR2-AN-2-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "2"]:
        """Residual function for the nonlinear equations."""
        x1, x2 = y

        # Problem parameter
        c = 0.01

        # G1: C * (X2 - sin(X1)) = 0
        res1 = c * (x2 - jnp.sin(x1))

        # G2: 2 * X1 = 0
        res2 = 2.0 * x1

        return jnp.array([res1, res2])

    def y0(self) -> Float[Array, "2"]:
        """Initial guess for the optimization problem."""
        return jnp.array([4.712389, -1.0])

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
