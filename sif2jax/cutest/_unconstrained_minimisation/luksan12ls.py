import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class LUKSAN12LS(AbstractUnconstrainedMinimisation):
    """Problem 12 (chained and modified HS47) in the paper L. Luksan: Hybrid methods in
    large sparse nonlinear least squares. J. Optimization Theory and Applications 89,
    pp. 575-595, 1996.

    SIF input: Nick Gould, June 2017.

    Classification: SUR2-AN-V-0

    Least-squares version
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    S: int = 32  # Seed for dimensions

    @property
    def n(self):
        """Number of variables: 3*S + 2."""
        return 3 * self.S + 2

    @property
    def y0(self):
        """Initial guess."""
        return jnp.full(self.n, -1.0)

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the least squares objective function.

        The objective is the sum of squares of M = 6*S equations.
        Each block of S generates 6 equations:
        - E(k): 10*x0^2 - 10*x1
        - E(k+1): x2 - 1.0
        - E(k+2): (x3 - 1)^2
        - E(k+3): (x4 - 1)^3
        - E(k+4): x3*x0^2 + sin(x3-x4) - 10.0
        - E(k+5): (x2^4)*(x3^2) + x1 - 20.0
        """
        del args  # Not used

        s = self.S

        # Vectorized computation
        # Extract variable indices for all blocks
        j_indices = jnp.arange(s)
        i_indices = 3 * j_indices  # Start indices for each block

        # Extract all variables for vectorized operations
        x0 = y[i_indices]  # X(I) for all blocks
        x1 = y[i_indices + 1]  # X(I+1) for all blocks
        x2 = y[i_indices + 2]  # X(I+2) for all blocks
        x3 = y[i_indices + 3]  # X(I+3) for all blocks
        x4 = y[i_indices + 4]  # X(I+4) for all blocks

        # Compute all residuals for each type
        # E(k): 10*x0^2 - 10*x1
        res1 = 10.0 * x0**2 - 10.0 * x1

        # E(k+1): x2 - 1.0
        res2 = x2 - 1.0

        # E(k+2): (x3 - 1)^2
        res3 = (x3 - 1.0) ** 2

        # E(k+3): (x4 - 1)^3
        res4 = (x4 - 1.0) ** 3

        # E(k+4): x3*x0^2 + sin(x3-x4) - 10.0
        res5 = x3 * x0**2 + jnp.sin(x3 - x4) - 10.0

        # E(k+5): (x2^4)*(x3^2) + x1 - 20.0
        res6 = x2**4 * x3**2 + x1 - 20.0

        # Stack all residuals in the correct order
        # For each block, we have 6 residuals
        residuals = jnp.stack([res1, res2, res3, res4, res5, res6], axis=1).flatten()

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
