import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class COSHFUN(AbstractConstrainedMinimisation):
    """
    COSHFUN problem in CUTEst.

    A nonlinear minmax problem.

    Source:
    K. Jonasson and K. Madsen,
    "Corrected sequential linear programming for sparse
    minimax optimization", Technical report, Institute for Numerical
    Analysis, Technical U. of Denmark.

    classification LOR2-AN-V-V

    SIF input: Nick Gould, October 1992.
    """

    # Number of functions
    m: int = 2000  # Default from SIF, can be 3, 8, 14, 20, 200, 2000

    @property
    def n(self) -> int:
        """Number of variables (3 * m)."""
        return 3 * self.m

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def initial_guess(self) -> jnp.ndarray:
        """Initial guess - all zeros (default when not specified in SIF)."""
        n = self.n
        y0 = jnp.zeros(n + 1, dtype=jnp.float64)  # n variables + f
        return y0

    @property
    def y0(self) -> jnp.ndarray:
        """Initial guess."""
        return self.initial_guess

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF)."""
        return None

    @property
    def bounds(self):
        """Variable bounds (unbounded)."""
        return None

    def objective(self, y: jnp.ndarray, args=None) -> jnp.ndarray:
        """Objective function - minimize f (the last variable)."""
        return y[-1]

    def constraint(self, y: jnp.ndarray):
        """Compute constraints."""
        # For a minmax problem, we have inequalities: c_i(x) <= f
        # Which we express as: c_i(x) - f <= 0
        inequalities = self.inequality_constraints(y)
        return None, inequalities

    def inequality_constraints(self, y: jnp.ndarray) -> jnp.ndarray:
        """Compute inequality constraints c_i(x) - f <= 0."""
        m = self.m
        n = self.n
        x = y[:-1]  # First n variables
        f = y[-1]  # Last variable is f

        # Vectorized indices for all constraints
        i_range = jnp.arange(m)

        # Indices for the nonlinear terms (vectorized)
        idx_sqr = 3 * i_range + 2  # X(3*i) in 1-based
        idx_cosh = 3 * i_range + 1  # X(3*i-1) in 1-based
        idx_prod1 = 3 * i_range  # X(3*i-2) in 1-based
        idx_prod2 = 3 * i_range + 2  # X(3*i) in 1-based

        # Nonlinear terms: x^2 + cosh(x) + 2*x*x*y (vectorized)
        c_vec = (
            x[idx_sqr] ** 2
            + jnp.cosh(x[idx_cosh])
            + 2.0 * x[idx_prod1] ** 2 * x[idx_prod2]
        )

        # Linear terms for the first constraint (C1)
        c_vec = c_vec.at[0].add(-2.0 * x[2] - x[5])

        # Linear terms for middle constraints
        # From SIF: C(I/3) has X(I-5), X(I), X(I+3) terms
        # where I goes from 6 to N-3 in steps of 3
        for i in range(1, m - 1):
            idx_base = 3 * (i + 1)  # This gives us 6, 9, 12, ...
            if idx_base < n - 2:
                c_vec = c_vec.at[i].add(
                    x[idx_base - 5] - 2.0 * x[idx_base] - x[idx_base + 3]
                )

        # Linear terms for the last constraint (C(M))
        c_vec = c_vec.at[m - 1].add(x[n - 6] - 2.0 * x[n - 1])

        # Return as c_i - f <= 0
        return c_vec - f
