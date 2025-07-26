import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SISSER2(AbstractUnconstrainedMinimisation):
    """A simple unconstrained problem in 2 variables.

    Source:
    F.S. Sisser,
    "Elimination of bounds in optimization problems by transforming
    variables",
    Mathematical Programming 20:110-121, 1981.

    See also Buckley#216 (p. 91)

    SIF input: Ph. Toint, Dec 1989.
    Modified version of SISSER (formulation corrected) May 2024

    classification OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self):
        """Number of variables."""
        return 2

    @property
    def y0(self):
        """Starting point."""
        return jnp.array([1.0, 0.1])

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Compute the objective function (vectorized).

        f(x) = (1/3 * x1^2)^2 + (-1/2 * x1*x2)^2 + (1/3 * x2^2)^2
        """
        del args
        x1, x2 = y[0], y[1]

        # Groups with their scales
        g1 = (1.0 / 3.0) * x1**2  # Scale 1/3, Element x1^2
        g2 = (-0.5) * x1 * x2  # Scale -1/2, Element x1*x2
        g3 = (1.0 / 3.0) * x2**2  # Scale 1/3, Element x2^2

        # Group type L2: squares each group value
        return g1**2 + g2**2 + g3**2

    @property
    def expected_result(self):
        """Expected solution: origin."""
        return jnp.array([0.0, 0.0])

    @property
    def expected_objective_value(self):
        """Expected optimal value is 0.0."""
        return jnp.array(0.0)
