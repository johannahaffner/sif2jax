"""A variation on Beale's problem in 2 variables.

Source: An adaptation by Ph. Toint of Problem 5 in
J.J. More', B.S. Garbow and K.E. Hillstrom,
"Testing Unconstrained Optimization Software",
ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

See also Buckley#89.
SIF input: Ph. Toint, Mar 2003.

Classification: SUR2-AN-V-0
"""

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class MODBEALE(AbstractUnconstrainedMinimisation):
    """A variation on Beale's problem."""

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Parameters
    n_half: int = 10000  # N/2 parameter from SIF
    ALPHA: float = 50.0

    @property
    def n(self):
        """Number of variables (2 * N/2)."""
        return 2 * self.n_half

    @property
    def y0(self):
        """Initial guess."""
        return jnp.ones(self.n)

    @property
    def args(self):
        """No additional arguments."""
        return None

    def objective(self, y, args):
        """Compute the objective function.

        The objective is the sum of squared residuals:
        - BA(i) = 1.5 - X(2i-1) * (1 - X(2i))
        - BB(i) = 2.25 - X(2i-1) * (1 - X(2i)^2)
        - BC(i) = 2.625 - X(2i-1) * (1 - X(2i)^3)
        - L(i) = (6*X(2i) - X(2i+1)) / ALPHA for i < N/2
        """
        del args  # Not used

        alphinv = 1.0 / self.ALPHA
        obj = 0.0

        # Process groups for i = 1 to N/2
        for i in range(self.n_half):
            j = 2 * i  # 2i-1 in 0-based indexing

            # PRODB elements with different powers
            x_j = y[j]
            x_j1 = y[j + 1]

            # BA(i) = 1.5 - X(2i-1) * (1 - X(2i))
            ba_res = 1.5 - x_j * (1.0 - x_j1)
            obj += ba_res**2

            # BB(i) = 2.25 - X(2i-1) * (1 - X(2i)^2)
            bb_res = 2.25 - x_j * (1.0 - x_j1**2)
            obj += bb_res**2

            # BC(i) = 2.625 - X(2i-1) * (1 - X(2i)^3)
            bc_res = 2.625 - x_j * (1.0 - x_j1**3)
            obj += bc_res**2

            # L(i) for i < N/2
            if i < self.n_half - 1:
                l_res = (6.0 * x_j1 - y[j + 2]) * alphinv
                obj += l_res**2

        return jnp.array(obj)

    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value is 0.0."""
        return jnp.array(0.0)
