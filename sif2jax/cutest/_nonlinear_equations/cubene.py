import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class CUBENE(AbstractNonlinearEquations):
    """
    A cubic variant of the Rosenbrock test function.
    Nonlinear equations version.

    Source: problem 5 (p. 89) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    classification NOR2-AN-2-2
    """

    n: int = 2  # Fixed dimension problem
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the cubic nonlinear equations"""
        # SQ(1): y(1) - 1 = 0
        res1 = y[0] - 1.0

        # SQ(2): 0.1 * (y(2) - y(1)^3) = 0
        res2 = 0.1 * (y[1] - y[0] ** 3)

        return jnp.array([res1, res2])

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return jnp.array([-1.2, 1.0], dtype=jnp.float64)

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # The solution is x = (1, 1)
        return jnp.array([1.0, 1.0], dtype=jnp.float64)

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
