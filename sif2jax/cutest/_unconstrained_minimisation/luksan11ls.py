import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class LUKSAN11LS(AbstractUnconstrainedMinimisation):
    """Problem 11 (chained serpentine) in the paper L. Luksan: Hybrid methods in
    large sparse nonlinear least squares. J. Optimization Theory and Applications 89,
    pp. 575-595, 1996.

    SIF input: Nick Gould, June 2017.

    Classification: SUR2-AN-V-0

    Least-squares version
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    S: int = 99  # Seed for dimensions

    @property
    def n(self):
        """Number of variables: S + 1."""
        return self.S + 1

    @property
    def y0(self):
        """Initial guess."""
        return jnp.full(self.n, -0.8)

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the least squares objective function.

        The objective is the sum of squares of M = 2*S equations:
        - For i = 1 to S:
          - E(2i-1) = 20*X(i)/(1+X(i)^2) - 10*X(i+1)
          - E(2i) = X(i) - RHS where RHS = 1.0 for E(2), E(4), E(6), ...
        """
        del args  # Not used

        s = self.S

        # Vectorized computation
        # Extract x[0:s] and x[1:s+1] for vectorized operations
        x_i = y[:-1]  # x[0] to x[s-1]
        x_ip1 = y[1:]  # x[1] to x[s]

        # First residuals: 20*x[i]/(1+x[i]^2) - 10*x[i+1]
        d = 1.0 + x_i * x_i
        res1 = 20.0 * x_i / d - 10.0 * x_ip1

        # Second residuals: x[i] - RHS
        # RHS = 1.0 for equations E(2), E(4), E(6), ... (1-based)
        # In 0-based these are residuals at indices 1, 3, 5, ...
        res2 = x_i.copy()

        # Create residuals array by interleaving res1 and res2
        residuals = jnp.zeros(2 * s)
        residuals = residuals.at[::2].set(res1)  # Even indices: 0, 2, 4, ...
        residuals = residuals.at[1::2].set(res2)  # Odd indices: 1, 3, 5, ...

        # Apply RHS: subtract 1.0 from equations E(2), E(4), E(6), ...
        # These are at 0-based indices 1, 3, 5, ...
        residuals = residuals.at[1::2].add(-1.0)

        # Sum of squares (L2 group type in SIF)
        return jnp.sum(residuals**2)

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
