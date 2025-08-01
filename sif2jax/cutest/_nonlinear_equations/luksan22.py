"""Problem 22 (attracting-repelling) in the paper L. Luksan: Hybrid methods in
large sparse nonlinear least squares. J. Optimization Theory and Applications 89,
pp. 575-595, 1996.

This is a large sparse nonlinear least squares problem with exponential terms,
designed to test optimization algorithms on problems with attracting-repelling
behavior due to the combination of exponential terms with different signs.

Source: Luksan, L. (1996)
Hybrid methods in large sparse nonlinear least squares
J. Optimization Theory and Applications 89, pp. 575-595.

SIF input: Nick Gould, June 1997.

Classification: NOR2-AN-V-V
"""

import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LUKSAN22(AbstractNonlinearEquations):
    """Luksan's problem 22 - attracting-repelling nonlinear equations."""

    n_var: int = 100  # Number of variables (default)
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n_var: int = 100):
        self.n_var = n_var

    @property
    def n(self) -> int:
        """Number of variables."""
        return self.n_var

    def num_residuals(self) -> int:
        """Number of residuals: 2*N - 2."""
        return 2 * self.n_var - 2

    def starting_point(self) -> Array:
        """Return the starting point for the problem."""
        # X(odd indices) = -1.2, X(even indices) = 1.0
        y0 = jnp.zeros(self.n)
        # Set odd indices (0, 2, 4, ...) to -1.2
        y0 = y0.at[::2].set(-1.2)
        # Set even indices (1, 3, 5, ...) to 1.0
        y0 = y0.at[1::2].set(1.0)
        return y0

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        The residuals are:
        - E(1) = X(1) + 1.0
        - For k=2 to 2N-3 step 2, i = (k+1)/2:
          - E(k) = 10.0 * (X(i)^2 - 10.0*X(i+1))
          - E(k+1) = 2*exp(-(X(i)-X(i+1))^2) - exp(-2*(X(i+1)-X(i+2))^2)
        - E(2N-2) = 10.0 * (X(N-1)^2 - 10.0*X(N))
        """
        residuals = []

        # E(1) = X(1) + 1.0
        e1 = y[0] + 1.0
        residuals.append(e1)

        # For k=2 to 2N-3 step 2
        # Note: we iterate from k=2 to 2N-3 (inclusive) with step 2
        # This gives us k = 2, 4, 6, ..., 2N-4
        for k in range(2, 2 * self.n_var - 2, 2):
            # i is 1-based in SIF, so i = (k+1)/2 gives i = 2, 3, ..., N-1
            # Convert to 0-based: i_idx = i - 1 = (k+1)/2 - 1 = (k-1)/2
            i_idx = (k - 1) // 2

            # E(k) = 10.0 * (X(i)^2 - 10.0*X(i+1))
            # Note: In SIF groups, E(k) gets coefficient 10.0 from GROUP USES
            ek = 10.0 * (y[i_idx] ** 2 - 10.0 * y[i_idx + 1])
            residuals.append(ek)

            # E(k+1) = 2*exp(-(X(i)-X(i+1))^2) - exp(-2*(X(i+1)-X(i+2))^2)
            # EXPDA: 2 * exp(-(X(i) - X(i+1))^2)
            # EXPDB: exp(-2*(X(i+1) - X(i+2))^2)
            term1 = 2.0 * jnp.exp(-((y[i_idx] - y[i_idx + 1]) ** 2))
            term2 = jnp.exp(-2.0 * ((y[i_idx + 1] - y[i_idx + 2]) ** 2))
            ek1 = term1 - term2
            residuals.append(ek1)

        # E(2N-2) = 10.0 * (X(N-1)^2 - 10.0*X(N))
        # Note: In SIF groups, E(2N-2) gets coefficient 10.0 from GROUP USES
        e_final = 10.0 * (y[self.n_var - 2] ** 2 - 10.0 * y[self.n_var - 1])
        residuals.append(e_final)

        return jnp.array(residuals)

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
        # Solution should satisfy residuals = 0
        return jnp.zeros(self.n)

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
        """Free bounds for all variables."""
        return None
