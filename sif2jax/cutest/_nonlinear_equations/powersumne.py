from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class POWERSUMNE(AbstractNonlinearEquations):
    """SCIPY global optimization benchmark example POWERSUM

    Fit: y = sum_j=1^n x_j^i

    Source:  Problem from the SCIPY benchmark set
      https://github.com/scipy/scipy/tree/master/benchmarks/ ...
              benchmarks/go_benchmark_functions

    Nonlinear-equation formulation of POWERSUM.SIF

    SIF input: Nick Gould, Jan 2020

    classification NOR2-MN-V-V
    """

    n: int = 4  # Default to n=4
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "4"]:
        """Residual function for the nonlinear equations."""
        x = y

        # Data values for n=4 case
        # Y(i) = sum_j=1^4 x_ref[j]^i where x_ref = [1, 2, 3, 2]
        x_ref = jnp.array([1.0, 2.0, 3.0, 2.0])

        # Calculate Y(i) = sum of x_ref[j]^i for i=1 to 4
        y_data = []
        for i in range(1, 5):  # i = 1, 2, 3, 4
            sum_val = jnp.sum(x_ref**i)
            y_data.append(sum_val)
        y_data = jnp.array(y_data)

        # Model: sum_j=1^n x_j^i for each i
        residuals = []
        for i in range(1, 5):  # i = 1, 2, 3, 4
            model = jnp.sum(x**i)
            residuals.append(model - y_data[i - 1])

        return jnp.array(residuals)

    def y0(self) -> Float[Array, "4"]:
        """Initial guess for the optimization problem."""
        return jnp.array([2.0, 2.0, 2.0, 2.0])

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "4"] | None:
        """Expected result of the optimization problem."""
        # Optimal solution from SIF file
        return jnp.array([1.0, 2.0, 3.0, 2.0])

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
