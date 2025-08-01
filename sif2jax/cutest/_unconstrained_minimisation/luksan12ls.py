"""Problem 12 (chained and modified HS47) in the paper L. Luksan: Hybrid methods in
large sparse nonlinear least squares. J. Optimization Theory and Applications 89,
pp. 575-595, 1996.

SIF input: Nick Gould, June 2017.

Classification: SUR2-AN-V-0

Least-squares version
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class LUKSAN12LS(AbstractUnconstrainedMinimisation):
    """Luksan's problem 12 - chained and modified HS47 least squares."""

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

        equations = []
        s = self.S

        # Process each block
        for j in range(s):
            # Index for variables
            i = 3 * j  # Start index for this block

            # Extract variables
            x0 = y[i]  # X(I) in SIF
            x1 = y[i + 1]  # X(I+1) in SIF
            x2 = y[i + 2]  # X(I+2) in SIF
            x3 = y[i + 3]  # X(I+3) in SIF
            x4 = y[i + 4]  # X(I+4) in SIF

            # E(k): 10*x0^2 - 10*x1
            res1 = 10.0 * x0**2 - 10.0 * x1
            equations.append(res1)

            # E(k+1): x2 - 1.0
            res2 = x2 - 1.0
            equations.append(res2)

            # E(k+2): (x3 - 1)^2
            res3 = (x3 - 1.0) ** 2
            equations.append(res3)

            # E(k+3): (x4 - 1)^3
            res4 = (x4 - 1.0) ** 3
            equations.append(res4)

            # E(k+4): x3*x0^2 + sin(x3-x4) - 10.0
            res5 = x3 * x0**2 + jnp.sin(x3 - x4) - 10.0
            equations.append(res5)

            # E(k+5): (x2^4)*(x3^2) + x1 - 20.0
            res6 = x2**4 * x3**2 + x1 - 20.0
            equations.append(res6)

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
