import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class JUDGENE(AbstractNonlinearEquations):
    """SCIPY global optimization benchmark example Judge.

    Fit: y  =  x_1 + a_i x_2 + b_i x_2^2 + e

    Source:  Problem from the SCIPY benchmark set
      https://github.com/scipy/scipy/tree/master/benchmarks/ ...
              benchmarks/go_benchmark_functions

    Nonlinear-equation formulation of JUDGE.SIF

    SIF input: Nick Gould, Jan 2020

    classification NOR2-MN-2-20
    """

    n: int = 2
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return jnp.array(20)

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        return jnp.array([1.0, 5.0])

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        Args:
            y: Array of shape (2,) containing [x1, x2]
            args: Additional arguments (unused)

        Returns:
            Array of shape (20,) containing the residuals
        """
        x1, x2 = y[0], y[1]

        # Data values
        a = jnp.array(
            [
                0.286,
                0.973,
                0.384,
                0.276,
                0.973,
                0.543,
                0.957,
                0.948,
                0.543,
                0.797,
                0.936,
                0.889,
                0.006,
                0.828,
                0.399,
                0.617,
                0.939,
                0.784,
                0.072,
                0.889,
            ]
        )

        b = jnp.array(
            [
                0.645,
                0.585,
                0.310,
                0.058,
                0.455,
                0.779,
                0.259,
                0.202,
                0.028,
                0.099,
                0.142,
                0.296,
                0.175,
                0.180,
                0.842,
                0.039,
                0.103,
                0.620,
                0.158,
                0.704,
            ]
        )

        y_data = jnp.array(
            [
                4.284,
                4.149,
                3.877,
                0.533,
                2.211,
                2.389,
                2.145,
                3.231,
                1.998,
                1.379,
                2.106,
                1.428,
                1.011,
                2.179,
                2.858,
                1.388,
                1.651,
                1.593,
                1.046,
                2.152,
            ]
        )

        # Model: x1 + a[i] * x2 + b[i] * x2^2
        model = x1 + a * x2 + b * (x2 * x2)

        # Residuals
        residuals = model - y_data

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
        # From the scipy benchmark, the solution is approximately:
        return jnp.array([0.86466, 1.2357])

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
