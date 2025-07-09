import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class BEALENE(AbstractNonlinearEquations):
    """
    Beale problem in 2 variables
    a nonlinear equation version.

    Source: Problem 5 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#89.
    SIF input: Ph. Toint, Dec 1989.

    classification NOR2-AN-2-3
    """

    n: int = 2
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.ones(self.n, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return 3

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the Beale problem"""
        x1, x2 = y

        # Residuals based on the element formulation: x1 * (1 - x2^pow) - constant
        # Group A: x1 * (1 - x2^1) - 1.5
        res1 = x1 * (1.0 - x2) - 1.5

        # Group B: x1 * (1 - x2^2) - 2.25
        res2 = x1 * (1.0 - x2**2) - 2.25

        # Group C: x1 * (1 - x2^3) - 2.625
        res3 = x1 * (1.0 - x2**3) - 2.625

        return jnp.array([res1, res2, res3])

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # The known solution for Beale's function is (3, 0.5)
        return jnp.array([3.0, 0.5], dtype=jnp.float64)

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
