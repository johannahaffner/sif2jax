import jax
import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LANCZOS2(AbstractNonlinearEquations):
    """NIST Data fitting problem LANCZOS2 given as an inconsistent set of
    nonlinear equations.

    Fit: y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x) + e

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Lanczos, C. (1956).
    Applied Analysis. Englewood Cliffs, NJ: Prentice Hall, pp. 272-280.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015
    classification NOR2-MN-6-24
    """

    n: int = 6
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1})

    def num_residuals(self) -> int:
        """Number of residuals."""
        return 24

    def starting_point(self, y0_id: int = 0) -> Array:
        """Return the starting point for the problem."""
        if y0_id == 0:
            # START1
            return jnp.array([1.2, 0.3, 5.6, 5.5, 6.5, 7.6])
        elif y0_id == 1:
            # START2
            return jnp.array([0.5, 0.7, 3.6, 4.2, 4.0, 6.3])
        else:
            raise ValueError(f"Invalid y0_id: {y0_id}")

    def _get_data(self):
        """Get the data points and values."""
        # X data points from SIF file (X1 to X24)
        x_data = jnp.array(
            [
                0.00e0,
                5.00e-2,
                1.00e-1,
                1.50e-1,
                2.00e-1,
                2.50e-1,
                3.00e-1,
                3.50e-1,
                4.00e-1,
                4.50e-1,
                5.00e-1,
                5.50e-1,
                6.00e-1,
                6.50e-1,
                7.00e-1,
                7.50e-1,
                8.00e-1,
                8.50e-1,
                9.00e-1,
                9.50e-1,
                1.00e0,
                1.05e0,
                1.10e0,
                1.15e0,
            ]
        )

        # Y data points from SIF file (Y1 to Y24)
        y_data = jnp.array(
            [
                2.51340e0,
                2.04433e0,
                1.66840e0,
                1.36642e0,
                1.12323e0,
                9.26890e-1,
                7.67934e-1,
                6.38878e-1,
                5.33784e-1,
                4.47936e-1,
                3.77585e-1,
                3.19739e-1,
                2.72013e-1,
                2.32497e-1,
                1.99659e-1,
                1.72270e-1,
                1.49341e-1,
                1.30070e-1,
                1.13812e-1,
                1.00042e-1,
                8.83321e-2,
                7.83354e-2,
                6.97669e-2,
                6.23931e-2,
            ]
        )

        return x_data, y_data

    def model(self, x, params):
        """Compute the model function: b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)"""
        b1, b2, b3, b4, b5, b6 = params
        return b1 * jnp.exp(-b2 * x) + b3 * jnp.exp(-b4 * x) + b5 * jnp.exp(-b6 * x)

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        Args:
            y: Array of shape (6,) containing [b1, b2, b3, b4, b5, b6]
            args: Additional arguments (unused)

        Returns:
            Array of shape (24,) containing the residuals
        """
        x_data, y_data = self._get_data()

        # Vectorized model computation using vmap
        y_pred = jax.vmap(lambda x: self.model(x, y))(x_data)

        # Residuals
        residuals = y_pred - y_data

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        return self.starting_point(self.y0_iD)

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self):
        """Expected result of the optimization problem."""
        # This is a data fitting problem - no predefined global solution provided
        return None

    @property
    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self):
        """Return the bounds for the variables."""
        # All variables are free according to the SIF file - no finite bounds
        return None
