import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class MARATOSB(AbstractUnconstrainedMinimisation):
    """A variant of the Maratos problem with penalty parameter = 0.000001.

    Source:
    Ph. Toint.

    SIF input: Ph. Toint, June 1990.

    classification OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return jnp.array(2)

    @property
    def y0(self):
        """Starting point."""
        return jnp.array([1.1, 0.1])

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Compute the objective function.

        f(x) = x1 + (1/p) * (x1^2 + x2^2 - 1)^2
        where p = 0.000001, so 1/p = 1000000
        """
        del args
        x1, x2 = y[0], y[1]

        # Penalty parameter: INVP means inverse penalty
        invp = 1.0 / 0.000001  # = 1000000

        # F group: x1
        f_term = x1

        # C group: invp * (x1^2 + x2^2 - 1)^2
        # Elements: x1^2 and x2^2
        # Group value before L2: x1^2 + x2^2 - 1
        # L2 group type: squares the group value
        constraint_val = x1**2 + x2**2 - 1.0
        c_term = invp * constraint_val**2

        return f_term + c_term

    @property
    def expected_result(self):
        """Expected solution is approximately (1, 0) for small penalty."""
        # With very small penalty, the constraint x1^2 + x2^2 = 1 is loosely enforced
        # The objective x1 is minimized subject to approximately x1^2 + x2^2 = 1
        # This gives x1 = -1, x2 = 0
        return jnp.array([-1.0, 0.0])

    @property
    def expected_objective_value(self):
        """Expected optimal value is approximately -1.0."""
        # At the solution (-1, 0), the objective is -1 + 0.000001 * 0 = -1
        return jnp.array(-1.0)
