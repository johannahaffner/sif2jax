import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class EGGCRATENE(AbstractNonlinearEquations):
    """
    SCIPY global optimization benchmark example EGGCRATE

    Fit: (x,y,5sinx,5siny) + e = 0

    Source:  Problem from the SCIPY benchmark set
        https://github.com/scipy/scipy/tree/master/benchmarks/ ...
                benchmarks/go_benchmark_functions

    Nonlinear-equation formulation of EGGCRATE.SIF

    SIF input: Nick Gould

    classification NOR2-MN-4-2
    """

    m: int = 4
    n: int = 2
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.array([1.0, 2.0], dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the egg crate problem"""
        x, y_var = y[0], y[1]

        # Initialize residuals
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        # F1: x
        residuals = residuals.at[0].set(x)

        # F2: y
        residuals = residuals.at[1].set(y_var)

        # F3: 5 * sin(x)
        residuals = residuals.at[2].set(5.0 * jnp.sin(x))

        # F4: 5 * sin(y)
        residuals = residuals.at[3].set(5.0 * jnp.sin(y_var))

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
        # For this problem, the solution would be where all residuals are zero
        # This occurs at x=0, y=0
        return jnp.zeros(self.n, dtype=jnp.float64)

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
