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

        # Define the matrix B
        b = jnp.zeros((p, p))
        k = 0.0
        for i in range(p):
            for j in range(p):
                k += 1.0
                b = b.at[i, j].set(jnp.sin(k * k))

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

    def y0(self) -> Float[Array, "4900"]:
        """Initial guess for the optimization problem."""
        p = self.p

        # Define the matrix B
        b = jnp.zeros((p, p))
        k = 0.0
        for i in range(p):
            for j in range(p):
                k += 1.0
                b = b.at[i, j].set(jnp.sin(k * k))

        # B(3,1) = 0.0 for p >= 3
        if p >= 3:
            b = b.at[2, 0].set(0.0)

        # Initial guess: X = B - 0.8 * sin(k^2) for each element
        x = jnp.zeros((p, p))
        k = 0.0
        for i in range(p):
            for j in range(p):
                k += 1.0
                sk2 = jnp.sin(k * k)
                x = x.at[i, j].set(b[i, j] - 0.8 * sk2)

        return x.flatten()

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Float[Array, "4900"] | None:
        """Expected result of the optimization problem."""
        # The SIF file doesn't provide explicit solution values
        return None

    def expected_objective_value(self) -> Float[Array, ""] | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)
