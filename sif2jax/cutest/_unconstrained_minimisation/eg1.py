import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class EG1(AbstractUnconstrainedMinimisation):
    """The EG1 function.

    A simple nonlinear problem given as an example in Section 1.2.3 of
    the LANCELOT Manual.

    Source:
    A.R. Conn, N. Gould and Ph.L. Toint,
    "LANCELOT, A Fortran Package for Large-Scale Nonlinear Optimization
    (Release A)" Springer Verlag, 1992.

    SIF input: N. Gould and Ph. Toint, June 1994.

    Classification: OBR2-AY-3-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 3  # Problem has 3 variables

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # GROUP1: x1^2
        f1 = x1**2

        # GROUP2: (x2*x3)^4
        f2 = (x2 * x3) ** 4

        # GROUP3: combination of sin(x2 + x1 + x3) and x1*x3
        f3_1 = x2 * jnp.sin(x2 + x1 + x3)
        f3_2 = x1 * x3

        return f1 + f2 + f3_1 + f3_2

    def y0(self):
        # Initial values from SIF file
        return jnp.array([1.0, 1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None
