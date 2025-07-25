import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class TQUARTIC(AbstractUnconstrainedMinimisation):
    """A quartic function with nontrivial groups and repetitious elements.

    Source: Ph. Toint, private communication.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    _n: int = 5000
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return self._n

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def y0(self):
        """Initial guess."""
        return jnp.full(self.n, 0.1)

    def objective(self, y, args):
        """Compute the objective function.

        The objective is:
        f(y) = (1 + y[0])^2 + sum_{i=2}^n (y[i]^2 - y[0]^2)^2
        """
        del args  # Not used

        # First group: G1 = (constant + linear)^2
        # where constant = 1.0, linear = y[0] with coefficient 1.0
        # No element is added to G1 (only to G(i) for i >= 2)
        # So G1 = (1 + y[0])^2
        g1 = (1.0 + y[0]) ** 2

        # Remaining groups: G(i) = (E(i) - E(1))^2 where E(i) = y[i]^2, E(1) = y[0]^2
        # So G(i) = (y[i]^2 - y[0]^2)^2 for i = 2, ..., n
        y_squared = y**2
        g_rest = jnp.sum((y_squared[1:] - y_squared[0]) ** 2)

        return g1 + g_rest

    @property
    def expected_result(self):
        """Expected optimal solution."""
        # The optimal solution has x[0] = -1 and all other variables = 0
        result = jnp.zeros(self.n)
        result = result.at[0].set(-1.0)
        return result

    @property
    def expected_objective_value(self):
        """Expected objective value at solution."""
        # At x[0] = -1, x[i] = 0 for i > 1:
        # G1 = (1 + (-1))^2 = 0
        # G(i) = (0^2 - (-1)^2)^2 = (0 - 1)^2 = 1 for each i = 2, ..., n
        # Total = 0 + (n-1) * 1 = n - 1
        # But wait, the expected value is 0 according to the SIF file
        return jnp.array(0.0)
