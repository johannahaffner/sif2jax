import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class EXP2NE(AbstractNonlinearEquations):
    """
    SCIPY global optimization benchmark example Exp2

    Fit: y = e^{-i/10 x_1} - 5e^{-i/10 x_2} - e^{-i/10} + 5e^{-i} + e

    Source:  Problem from the SCIPY benchmark set
      https://github.com/scipy/scipy/tree/master/benchmarks/ ...
              benchmarks/go_benchmark_functions

    Nonlinear-equation formulation of EXP2.SIF

    SIF input: Nick Gould

    classification NOR2-MN-2-10
    """

    n: int = 2
    m: int = 10
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return self.m

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.array([1.0, 5.0])

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        The problem has residuals of the form:
        F(i) = e^{-i/10 x_1} - 5e^{-i/10 x_2} - y_i

        where y_i = e^{-i/10} - 5e^{-i}
        """
        x1, x2 = y[0], y[1]
        residuals = jnp.zeros(self.m, dtype=jnp.float64)

        for i in range(self.m):
            # Compute the target value
            e1 = jnp.exp(-i / 10.0)
            e2 = jnp.exp(-float(i))
            y_i = e1 - 5.0 * e2

            # Compute the residual
            exp1 = jnp.exp(-i / 10.0 * x1)
            exp2 = jnp.exp(-i / 10.0 * x2)
            residuals = residuals.at[i].set(exp1 - 5.0 * exp2 - y_i)

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
        # Solution is x1 = 1.0, x2 = 10.0
        return jnp.array([1.0, 10.0])

    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """Free bounds for all variables."""
        return None
