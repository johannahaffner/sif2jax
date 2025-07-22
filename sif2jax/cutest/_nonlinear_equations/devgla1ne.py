import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class DEVGLA1NE(AbstractNonlinearEquations):
    """
    SCIPY global optimization benchmark example DeVilliersGlasser01

    Fit: y  = x_1 x_2^t sin( t x_3 + x_4 )  +  e

    Source:  Problem from the SCIPY benchmark set
        https://github.com/scipy/scipy/tree/master/benchmarks/ ...
                benchmarks/go_benchmark_functions

    Nonlinear-equation formulation of DEVGLA1.SIF

    SIF input: Nick Gould, Jan 2020

    classification NOR2-MN-4-24
    """

    m: int = 24
    n: int = 4
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def starting_point(self) -> Array:
        return jnp.full(4, 2.0, dtype=jnp.float64)

    def num_residuals(self) -> int:
        return self.m

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals of the DeVilliersGlasser01 problem"""
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]

        # Precompute data values
        a = 1.371
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        for i in range(self.m):
            t = i * 0.1  # t = (i-1) * 0.1 in 0-based indexing
            at = a**t
            tp = t * 3.112
            tpa = tp + 1.761
            stpa = jnp.sin(tpa)
            p = at * stpa
            pp = p * 60.137
            y_i = pp

            # Element DG1: x1 * x2^t * sin(t * x3 + x4)
            f_i = x1 * (x2**t) * jnp.sin(t * x3 + x4)
            residuals = residuals.at[i].set(f_i - y_i)

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Solution is not provided in the SIF file
        return self.starting_point()

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
