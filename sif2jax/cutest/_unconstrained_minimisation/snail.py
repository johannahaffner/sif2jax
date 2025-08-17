import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SNAIL(AbstractUnconstrainedMinimisation):
    """A 2D problem featuring a spiraling valley.

    Dedicated to the city of Namur, whose emblem is a snail.

    Source: J. Engels, private communication.
    SIF input: Ph. Toint, May 1990.

    Classification: OUR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # Fixed to 2 variables

    # Problem parameters
    clow: float = 1.0  # Lower spiral parameter
    cup: float = 2.0  # Upper spiral parameter (depth of valley is cup - clow)

    def __init__(self, clow: float = 1.0, cup: float = 2.0):
        """Initialize SNAIL problem.

        Args:
            clow: Lower spiral parameter (default 1.0)
            cup: Upper spiral parameter (default 2.0)
        """
        self.clow = clow
        self.cup = cup
        self.n = 2

    def objective(self, y, args):
        """Compute objective function.

        The objective involves a spiral element with complex trigonometric calculations.
        """
        del args
        x1, x2 = y[0], y[1]

        # Parameters
        a = 0.5 * (self.cup + self.clow)
        b = 0.5 * (self.cup - self.clow)

        # Compute intermediate values
        x2_sq = x1 * x1
        y2_sq = x2 * x2
        r2 = x2_sq + y2_sq

        # Avoid division by zero for r
        r = jnp.sqrt(r2 + 1e-10)  # Add small epsilon for numerical stability

        # First part: u = r2 / (1 + r2)
        d = 1.0 + r2
        u = r2 / d

        # Second part involves spiral: v = 1 + a*r - r*b*cos(r - theta)
        # where theta = atan2(x2, x1)
        theta = jnp.arctan2(x2, x1)
        arg = r - theta
        c = b * jnp.cos(arg)
        v = 1.0 + a * r - r * c

        # Objective function
        f = u * v

        return f

    @property
    def bounds(self):
        """All variables are unbounded."""
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        return lower, upper

    @property
    def y0(self):
        """Starting point: x1=10, x2=10."""
        return jnp.array([10.0, 10.0])

    @property
    def args(self):
        """No additional arguments."""
        return None

    @property
    def expected_result(self):
        """Expected solution is near the origin."""
        # The spiral valley leads to a minimum near (0, 0)
        return jnp.zeros(self.n)

    @property
    def expected_objective_value(self):
        """Expected minimum value is 0."""
        return jnp.array(0.0)
