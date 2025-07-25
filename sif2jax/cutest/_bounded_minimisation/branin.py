import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class BRANIN(AbstractBoundedMinimisation):
    """The Branin & Ho function - a global optimization test problem.

    The Branin function has the form:
    a(x2 - b*x1**2 + c*x1 - r)**2 + s(1-t)cos(x1) + s
    where a=1, b=5.1/(4π²), c=5/π, s=10 and t=1/(8π),
    and x1 in [-5, 10], x2 in [0, 15]

    Source:
    L. C. W. Dixon and G. P. Szego (Eds.)
    Towards Global Optimization
    North Holland, 1975.

    SIF input: Nick Gould, July 2021

    classification: OBR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # 2 variables

    @property
    def y0(self):
        # Starting point from SIF file
        return jnp.array([5.0, 10.0])

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Compute the objective function.

        Minimize: a(x2 - b*x1**2 + c*x1 - r)**2 + s(1-t)cos(x1) + s
        """
        x1, x2 = y

        # Parameters
        pi = jnp.pi
        a = 1.0
        b = 5.1 / (4.0 * pi**2)
        c = 5.0 / pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * pi)

        # First term: a(x2 - b*x1**2 + c*x1 - r)**2
        inner = x2 - b * x1**2 + c * x1 - r
        term1 = a * inner**2

        # Second term: s(1-t)cos(x1) + s
        term2 = s * (1.0 - t) * jnp.cos(x1) + s

        return term1 + term2

    @property
    def bounds(self):
        """Returns the bounds on the variables."""
        # From SIF file: x1 in [-5, 10], x2 in [0, 15]
        lower = jnp.array([-5.0, 0.0])
        upper = jnp.array([10.0, 15.0])
        return lower, upper

    @property
    def expected_result(self):
        # The Branin function has three global minima:
        # (-π, 12.275), (π, 2.275), (9.42478, 2.475)
        # Return one of them
        return jnp.array([-jnp.pi, 12.275])

    @property
    def expected_objective_value(self):
        # The global minimum value
        return jnp.array(0.397887)
