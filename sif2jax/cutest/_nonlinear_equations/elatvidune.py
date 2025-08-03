import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class ELATVIDUNE(AbstractNonlinearEquations):
    """
    SCIPY global optimization benchmark example ElAttarVidyasagarDutta

    Fit: (x_1^2 + x_2 − 10, x_1 + x_2^2 − 7, x_1^2 + x_2^3 − 1) + e = 0

    Source:  Problem from the SCIPY benchmark set
        https://github.com/scipy/scipy/tree/master/benchmarks/ ...
                benchmarks/go_benchmark_functions

    Nonlinear-equation formulation of EGGCRATE.SIF

    SIF input: Nick Gould

    classification NOR2-MN-2-3
    """

    m: int = 3
    n: int = 2
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.array([1.0, 5.0], dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the ElAttarVidyasagarDutta problem"""
        x1, x2 = y[0], y[1]

        # Initialize residuals
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        # F1: x1^2 + x2 - 10
        residuals = residuals.at[0].set(x1 * x1 + x2 - 10.0)

        # F2: x1 + x2^2 - 7
        residuals = residuals.at[1].set(x1 + x2 * x2 - 7.0)

        # F3: x1^2 + x2^3 - 1
        residuals = residuals.at[2].set(x1 * x1 + x2**3 - 1.0)

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Solution is not provided in the SIF file
        return self.starting_point()

    @property
    def expected_objective_value(self) -> Array:
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
