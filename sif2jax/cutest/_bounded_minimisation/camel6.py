import jax.numpy as jnp

from ..._problem import AbstractBoundedMinimisation


class CAMEL6(AbstractBoundedMinimisation):
    """The Six Hump Camel function.

    A test function for global optimization algorithms with 6 local minima,
    2 of which are global minima.

    The objective function is:
    f(x) = 4x₁² - 2.1x₁⁴ + (1/3)x₁⁶ + x₁x₂ - 4x₂² + 4x₂⁴

    Source:
    L. C. W. Dixon and G. P. Szego (Eds.)
    Towards Global Optimization
    North Holland, 1975.

    SIF input: A.R. Conn May 1995

    classification: OBR2-AN-2-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 2  # 2 variables

    @property
    def y0(self):
        """Starting point from SIF file."""
        return jnp.array([1.1, 1.1])

    @property
    def args(self):
        return ()

    def objective(self, y, args):
        """Compute the objective function.

        f(x) = 4x₁² - 2.1x₁⁴ + (1/3)x₁⁶ + x₁x₂ - 4x₂² + 4x₂⁴
        """
        x1, x2 = y

        # Element values
        e1 = x1**2  # SQ element
        e2 = x1**4  # FORPW element
        e3 = x1**6  # SIXPW element
        e4 = x1 * x2  # 2PROD element
        e5 = x2**2  # SQ element
        e6 = x2**4  # FORPW element

        # Objective function from GROUP USES
        # E  OBJ       E1        4.0            E2        -2.1
        # E  OBJ       E3        0.333333333333 E4
        # E  OBJ       E5        -4.0           E6        4.0
        obj = 4.0 * e1 - 2.1 * e2 + (1.0 / 3.0) * e3 + e4 - 4.0 * e5 + 4.0 * e6

        return obj

    @property
    def bounds(self):
        """Returns the bounds on the variables."""
        # From SIF file:
        # UP CAMEL6    X1        3.0
        # LO CAMEL6    X1        -3.0
        # UP CAMEL6    X2        1.5
        # LO CAMEL6    X2        -1.5
        lower = jnp.array([-3.0, -1.5])
        upper = jnp.array([3.0, 1.5])
        return lower, upper

    @property
    def expected_result(self):
        # The six hump camel has two global minima
        # at approximately (0.0898, -0.7126) and (-0.0898, 0.7126)
        return jnp.array([0.0898, -0.7126])

    @property
    def expected_objective_value(self):
        # The global minimum value
        return jnp.array(-1.0316)
