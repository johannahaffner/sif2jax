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

Classification: SUR2-AN-V-0
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class LUKSAN22LS(AbstractUnconstrainedMinimisation):
    """Luksan's problem 22 - attracting-repelling least squares."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    N: int = 100  # Number of variables (default)

    @property
    def n(self):
        """Number of variables."""
        return self.N

    @property
    def y0(self):
        """Initial guess."""
        # X(odd indices) = -1.2, X(even indices) = 1.0
        y0 = jnp.zeros(self.n)
        # Set odd indices (0, 2, 4, ...) to -1.2
        y0 = y0.at[::2].set(-1.2)
        # Set even indices (1, 3, 5, ...) to 1.0
        y0 = y0.at[1::2].set(1.0)
        return y0

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the least squares objective function.

        The objective is the sum of squares of 2*N-2 equations:
        - E(1) = X(1) + 1.0
        - For k=2 to 2N-3 step 2, i = (k+1)/2:
          - E(k) = 10.0 * (X(i)^2 - 10.0*X(i+1))
          - E(k+1) = 2*exp(-(X(i)-X(i+1))^2) - exp(-2*(X(i+1)-X(i+2))^2)
        - E(2N-2) = 10.0 * (X(N-1)^2 - 10.0*X(N))
        """
        del args  # Not used

        equations = []

        # E(1) = X(1) + 1.0
        e1 = y[0] + 1.0
        equations.append(e1)

        # For k=2 to 2N-3 step 2, i = (k+1)/2
        for k in range(2, 2 * self.N - 2, 2):
            i = (k + 1) // 2 - 1  # Convert to 0-based indexing

            # E(k) = 10.0 * (X(i)^2 - 10.0*X(i+1))
            ek = 10.0 * (y[i] ** 2 - 10.0 * y[i + 1])
            equations.append(ek)

            # E(k+1) = 2*exp(-(X(i)-X(i+1))^2) - exp(-2*(X(i+1)-X(i+2))^2)
            if i + 2 < self.N:  # Make sure X(i+2) exists
                term1 = 2.0 * jnp.exp(-((y[i] - y[i + 1]) ** 2))
                term2 = jnp.exp(-2.0 * ((y[i + 1] - y[i + 2]) ** 2))
                ek1 = term1 - term2
                equations.append(ek1)

        # E(2N-2) = 10.0 * (X(N-1)^2 - 10.0*X(N))
        e_final = 10.0 * (y[self.N - 2] ** 2 - 10.0 * y[self.N - 1])
        equations.append(e_final)

        # Sum of squares
        equations_array = jnp.array(equations)
        return jnp.sum(equations_array**2)

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None
