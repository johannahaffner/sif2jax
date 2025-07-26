import equinox as eqx
import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class QUARTC(AbstractUnconstrainedMinimisation):
    """A simple quartic function.

    Source: problem 157 (p. 87) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, March 1991.

    classification OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Number of variables
    n: int = eqx.field(default=5000, init=False)

    @property
    def y0(self):
        """Starting point."""
        return jnp.full(self.n, 2.0)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Compute the objective function (vectorized).

        f(x) = sum_{i=1}^n (x_i - i)^4
        """
        del args

        # Create indices 1 to n
        i_vals = jnp.arange(1, self.n + 1, dtype=jnp.float64)

        # Compute (x_i - i)^4 for all i
        diff = y - i_vals
        quartic_terms = diff**4

        return jnp.sum(quartic_terms)

    @property
    def expected_result(self):
        """Expected solution: x_i = i for all i."""
        return jnp.arange(1, self.n + 1, dtype=jnp.float64)

    @property
    def expected_objective_value(self):
        """Expected optimal value is 0.0."""
        return jnp.array(0.0)
