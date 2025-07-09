import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class ARGAUSS(AbstractNonlinearEquations):
    """
    More's gaussian problem in 3 variables

    Source: Problem 9 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#28
    SIF input: Ph. Toint, Dec 1989.

    classification NOR2-AN-3-15
    """

    n: int = 3
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.array([0.4, 1.0, 0.0], dtype=jnp.float64)

    def num_residuals(self) -> int:
        return 15

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the gaussian problem"""
        x1, x2, x3 = y

        # Constants from the SIF file
        y_data = jnp.array(
            [
                0.0009,
                0.0044,
                0.0175,
                0.0540,
                0.1295,
                0.2420,
                0.3521,
                0.3989,
                0.3521,
                0.2420,
                0.1295,
                0.0540,
                0.0175,
                0.0044,
                0.0009,
            ],
            dtype=jnp.float64,
        )

        # Compute residuals
        residuals = []
        for i in range(15):
            t = 0.5 * (8.0 - (i + 1.0))
            tmv3 = t - x3
            tmv3sq = -0.5 * tmv3 * tmv3
            expa = jnp.exp(x2 * tmv3sq)
            fval = x1 * expa
            residuals.append(fval - y_data[i])

        return jnp.array(residuals)

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # Solution from CUTEst documentation
        return jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
