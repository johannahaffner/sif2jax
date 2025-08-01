"""Problem 11 (chained serpentine) in the paper L. Luksan: Hybrid methods in
large sparse nonlinear least squares. J. Optimization Theory and Applications 89,
pp. 575-595, 1996.

SIF input: Nick Gould, June 2017.

Classification: SUR2-AN-V-0

Least-squares version
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class LUKSAN11LS(AbstractUnconstrainedMinimisation):
    """Luksan's problem 11 - chained serpentine least squares."""

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
          - E(2i) = X(i) [with RHS = 1.0 for odd i]
        """
        del args  # Not used

        equations = []
        s = self.S

        # Build residuals in pairs
        for i in range(s):
            # First residual of pair: 20*x[i]/(1+x[i]^2) - 10*x[i+1]
            xi = y[i]
            d = 1.0 + xi * xi
            res1 = 20.0 * xi / d - 10.0 * y[i + 1]
            equations.append(res1)

            # Second residual of pair: x[i] - c
            # where c = 1.0 for odd equation indices (E(2), E(4), E(6), ...)
            # In 0-based indexing, these are equations at indices 1, 3, 5, ...
            # Since we're at iteration i, this is the (2*i+2)-th equation
            # For i=0: E(2), for i=1: E(4), etc.
            # So c = 1.0 when i is even (0, 2, 4, ...)
            if i % 2 == 0:
                res2 = y[i] - 1.0
            else:
                res2 = y[i]
            equations.append(res2)

        # Sum of squares (L2 group type in SIF)
        equations_array = jnp.array(equations)
        return jnp.sum(equations_array**2)

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return jnp.array(0.0)
