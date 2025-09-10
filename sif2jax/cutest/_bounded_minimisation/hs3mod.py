import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class HS3MOD(AbstractBoundedMinimisation):
    """Hock and Schittkowski problem 3 (Modified version).

    Source: problem 3 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187, Springer
    Verlag, Heidelberg, 1981.

    SIF input: A.R. Conn March 1990

    classification QBR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 2

    @property
    def y0(self):
        """Initial guess."""
        return jnp.array([10.0, 1.0], dtype=jnp.float64)

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Objective function."""
        del args
        x1, x2 = y[0], y[1]

        # G1: x2
        g1 = x2

        # G2: (-x1 + x2)^2 with L2 group type
        g2 = (-x1 + x2) * (-x1 + x2)

        return g1 + g2

    @property
    def bounds(self):
        """Lower bound on x2."""
        lower = jnp.array([-jnp.inf, 0.0], dtype=jnp.float64)
        upper = jnp.array([jnp.inf, jnp.inf], dtype=jnp.float64)
        return lower, upper

    @property
    def expected_result(self):
        """Expected result not provided in SIF file."""
        return None

    @property
    def expected_objective_value(self):
        """Expected objective value from SIF file."""
        # LO SOLTN = 0.0
        return jnp.array(0.0)
