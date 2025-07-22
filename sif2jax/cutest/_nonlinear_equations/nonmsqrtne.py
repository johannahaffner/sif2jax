from jax import numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractNonlinearEquations


class NONMSQRTNE(AbstractNonlinearEquations):
    """The "non-matrix square root problem" obtained from an error in
    writing a correct matrix square root problem B by Nocedal and Liu.
    This is a nonlinear equation variant of NONMSQRT

    Source:
    Ph. Toint

    SIF input: Ph. Toint, Dec 1989.
               Nick Gould (nonlinear equation version), Jan 2019

    classification NOR2-AN-V-V
    """

    p: int = 70  # Default to p=70 (n=4900 variables)
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def residual(self, y, args) -> Float[Array, "4900"]:
        """Residual function for the nonlinear equations."""
        p = self.p

        # Reshape y to p x p matrix X
        x = y.reshape((p, p))

        # Define the matrix B using vectorized operations
        # Create k values: 1, 2, 3, ..., p*p
        k_values = jnp.arange(1, p * p + 1, dtype=float)
        # Compute sin(k^2) for all values
        sin_k2 = jnp.sin(k_values * k_values)
        # Reshape to p x p matrix
        b = sin_k2.reshape((p, p))

        # B(3,1) = 0.0 for p >= 3
        if p >= 3:
            b = b.at[2, 0].set(0.0)

        # Compute A = B * B
        a = b @ b

        # Compute X * X
        x_squared = x @ x

        # Residual: X * X - A
        residual = x_squared - a

        # Flatten to 1D array
        return residual.flatten()

    @property
    def y0(self) -> Float[Array, "4900"]:
        """Initial guess for the optimization problem."""
        p = self.p

        # Define the matrix B using vectorized operations
        # Create k values: 1, 2, 3, ..., p*p
        k_values = jnp.arange(1, p * p + 1, dtype=float)
        # Compute sin(k^2) for all values
        sin_k2 = jnp.sin(k_values * k_values)
        # Reshape to p x p matrix
        b = sin_k2.reshape((p, p))

        # B(3,1) = 0.0 for p >= 3
        if p >= 3:
            b = b.at[2, 0].set(0.0)

        # Initial guess: X = B - 0.8 * sin(k^2) for each element
        # sin_k2 already contains sin(k^2) values
        x = b - 0.8 * sin_k2.reshape((p, p))

        return x.flatten()

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self) -> None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide explicit solution values
        return None

    @property
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
