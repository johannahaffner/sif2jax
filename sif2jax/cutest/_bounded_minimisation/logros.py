import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class LOGROS(AbstractBoundedMinimisation):
    """A logarithmic rescaled variant of the old Rosenbrock's function.

    Source:
    Ph. Toint, private communication, 1991.

    SDIF input: Ph. Toint, June 1993.

    classification OBR2-AN-2-0
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
        return jnp.array([-1.2, 1.0])

    @property
    def args(self):
        return None

    def objective(self, y, args):
        """Compute the objective function.

        f(x,y) = log(1 + ROS)
        where ROS = 10000*(y - x^2)^2 + (1-x)^2
        """
        del args
        x, y_var = y[0], y[1]

        # Rosenbrock element with scaling
        w = 10000.0
        t = y_var - x * x
        ros = w * t**2 + (1.0 - x) ** 2

        # LOG group type: log(1 + GVAR)
        return jnp.log(1.0 + ros)

    @property
    def bounds(self):
        """Both variables have lower bounds of 0."""
        return (jnp.zeros(2), jnp.full(2, jnp.inf))

    @property
    def expected_result(self):
        """Expected solution: the Rosenbrock minimum at (1, 1)."""
        return jnp.array([1.0, 1.0])

    @property
    def expected_objective_value(self):
        """Expected optimal value is log(1) = 0."""
        return jnp.array(0.0)
