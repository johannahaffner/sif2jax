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
        # Create powers array: i = 1, 2, 3, 4
        powers = jnp.arange(1, 5, dtype=float)[:, None]  # Shape: (4, 1)

        # Compute x_ref^i for all i using broadcasting
        # Shape: (4, 4) where [i, j] = x_ref[j]^powers[i]
        x_ref_powers = x_ref[None, :] ** powers  # Shape: (4, 4)

        # Sum over j axis to get Y(i)
        y_data = jnp.sum(x_ref_powers, axis=1)  # Shape: (4,)

        # Model: sum_j=1^n x_j^i for each i
        # Similarly compute x^i for all i
        x_powers = x[None, :] ** powers  # Shape: (4, 4)

        # Sum over j axis to get model values
        model = jnp.sum(x_powers, axis=1)  # Shape: (4,)

        # Compute residuals
        residuals = model - y_data

        return residuals

    @property
    def y0(self) -> Float[Array, "4"]:
        """Initial guess for the optimization problem."""
        return jnp.array([2.0, 2.0, 2.0, 2.0])

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "4"]:
        """Expected result of the optimization problem."""
        # Optimal solution from SIF file
        return jnp.array([1.0, 2.0, 3.0, 2.0])

    def expected_objective_value(self) -> Float[Array, ""]:
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
