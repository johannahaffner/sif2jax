import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class LUKSAN22(AbstractNonlinearEquations):
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

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n_var: int = 100  # Number of variables (default)

    @property
    def n(self) -> int:
        """Number of variables."""
        return self.n_var

    @property
    def m(self) -> int:
        """Number of residuals: 2*N - 2."""
        return 2 * self.n_var - 2

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector.

        The residuals are:
        - E(1) = X(1) - 1.0 (note: pycutest uses -1.0, SIF file shows +1.0)
        - For k=2 to 2N-3 step 2, i = (k+1)/2:
          - E(k) = 10.0 * X(i)^2 - 10.0*X(i+1)
          - E(k+1) = 2*exp(-(X(i)-X(i+1))^2) + exp(-2*(X(i+1)-X(i+2))^2)
            (note: pycutest uses + between exponentials)
        - E(2N-2) = 10.0 * X(N-1)^2
        """
        del args  # Not used

        n = self.n_var

        # Initialize residuals array
        residuals = jnp.zeros(self.m)

        # E(1) = X(1) - 1.0 (first residual)
        # Note: SIF file shows +1.0 constant but pycutest uses -1.0
        residuals = residuals.at[0].set(y[0] - 1.0)

        # Vectorize the loop: k ranges from 2 to 2N-3 step 2
        # This gives k = 2, 4, 6, ..., 2N-4
        # Number of pairs: (2N-4 - 2)/2 + 1 = N-2 pairs
        num_pairs = n - 2
        k_vals = 2 + 2 * jnp.arange(num_pairs)  # k = 2, 4, 6, ..., 2N-4
        i_idx_vals = (k_vals - 1) // 2  # i_idx = 1, 2, 3, ..., N-2 (0-based)

        # Vectorized computation of E(k) = 10.0 * X(i)^2 - 10.0*X(i+1)
        ek_vals = 10.0 * y[i_idx_vals] ** 2 - 10.0 * y[i_idx_vals + 1]

        # Vectorized computation of E(k+1)
        # Note: pycutest uses + instead of - between exponential terms
        term1_vals = 2.0 * jnp.exp(-((y[i_idx_vals] - y[i_idx_vals + 1]) ** 2))
        term2_vals = jnp.exp(-2.0 * ((y[i_idx_vals + 1] - y[i_idx_vals + 2]) ** 2))
        ek1_vals = term1_vals + term2_vals

        # Interleave ek_vals and ek1_vals into residuals starting from index 1
        # We need to place them at indices 1, 2, 3, 4, 5, 6, ..., 2N-3
        even_indices = k_vals - 1  # Indices for E(k): 1, 3, 5, ..., 2N-3
        odd_indices = k_vals  # Indices for E(k+1): 2, 4, 6, ..., 2N-2

        residuals = residuals.at[even_indices].set(ek_vals)
        residuals = residuals.at[odd_indices].set(ek1_vals)

        # E(2N-2) = 10.0 * X(N-1)^2 (final residual)
        residuals = residuals.at[-1].set(10.0 * y[n - 2] ** 2)

        return residuals

    @property
    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        # X(odd indices) = -1.2, X(even indices) = 1.0
        y0 = jnp.zeros(self.n)
        # Set odd indices (0, 2, 4, ...) to -1.2
        y0 = y0.at[::2].set(-1.2)
        # Set even indices (1, 3, 5, ...) to 1.0
        y0 = y0.at[1::2].set(1.0)
        return y0

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
