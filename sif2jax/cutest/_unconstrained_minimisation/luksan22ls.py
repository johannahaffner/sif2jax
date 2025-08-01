import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class LUKSAN22LS(AbstractUnconstrainedMinimisation):
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

        The objective is the sum of squares of M = 2*N-2 equations:
        - E(1) = X(1) + 1.0
        - For k=2 to 2N-3 step 2, i = (k+1)/2:
          - E(k) = 10.0 * (X(i)^2 - 10.0*X(i+1))
          - E(k+1) = 2*exp(-(X(i)-X(i+1))^2) + exp(-2*(X(i+1)-X(i+2))^2)
        - E(2N-2) = 10.0 * (X(N-1)^2 - 10.0*X(N))
        """
        del args  # Not used

        n = self.N

        # Vectorized computation
        # E(1) = X(1) + 1.0
        e1 = y[0] + 1.0

        # For the middle equations, we have pairs of equations
        # i ranges from 1 to n-2 (0-based: 0 to n-3)
        i_indices = jnp.arange(n - 2)

        # Extract variables for vectorized operations
        xi = y[i_indices]  # X(i) for i=1 to n-2
        xi1 = y[i_indices + 1]  # X(i+1)
        xi2 = y[i_indices + 2]  # X(i+2)

        # E(k) equations: 10.0 * (X(i)^2 - 10.0*X(i+1))
        ek_vals = 10.0 * (xi**2 - 10.0 * xi1)

        # Vectorized computation of E(k+1)
        # E(k+1) = 2*exp(-(X(i)-X(i+1))^2) + exp(-2*(X(i+1)-X(i+2))^2)
        term1 = 2.0 * jnp.exp(-((xi - xi1) ** 2))
        term2 = jnp.exp(-2.0 * ((xi1 - xi2) ** 2))
        ek1_vals = term1 + term2

        # E(2N-2) = 10.0 * (X(N-1)^2 - 10.0*X(N))
        e_final = 10.0 * (y[n - 2] ** 2 - 10.0 * y[n - 1])

        # Combine all equations
        # We need to interleave ek_vals and ek1_vals
        # Create array with space for all equations
        equations = jnp.zeros(2 * n - 2)

        # Set E(1)
        equations = equations.at[0].set(e1)

        # Set the middle equations by interleaving
        # E(2), E(4), E(6), ... are ek_vals[0], ek_vals[1], ...
        # E(3), E(5), E(7), ... are ek1_vals[0], ek1_vals[1], ...
        equations = equations.at[1 : 2 * n - 3 : 2].set(ek_vals)
        equations = equations.at[2 : 2 * n - 3 : 2].set(ek1_vals)

        # Set E(2N-2)
        equations = equations.at[-1].set(e_final)

        # Sum of squares (L2 group type in SIF)
        return jnp.sum(equations**2)

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None
