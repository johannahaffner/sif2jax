import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class FLETCBV2(AbstractUnconstrainedMinimisation):
    """The FLETCBV2 function.

    Another Boundary Value problem.

    Source: The first problem given by
    R. Fletcher, "An optimal positive definite update for sparse Hessian matrices"
    Numerical Analysis report NA/145, University of Dundee, 1992.
    but assuming that the 1/h**2 term should read h**2
    This is what Fletcher intended (private communication).

    The author comments: "The problem arises from discretizing the bvp
                 x"=-2+sin x in [0,1]
    with x(0)=0, x(1)=1. This gives a symmetric system of equations,
    the residual vector of which is the gradient of the given function."
    He multiplies through by h^2 before integrating.

    SIF input: Nick Gould, Nov 1992.
    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Default dimension in SIF file
    kappa: float = 1.0  # Parameter used in the problem

    def objective(self, y, args):
        del args
        h = 1.0 / (self.n + 1)
        h2 = h * h

        # Define components as described in the SIF file
        # G(0): 0.5 * (x_1)^2
        f1 = 0.5 * (y[0]) ** 2

        # G(i) for i=1...n-1: 0.5 * (x_i - x_{i+1})^2
        f2 = 0.5 * jnp.sum((y[:-1] - y[1:]) ** 2)

        # G(n): 0.5 * (x_n)^2
        f3 = 0.5 * (y[-1]) ** 2

        # L(i) for i=1...n-1: x_i * (-2*h2)
        f4 = -2.0 * h2 * jnp.sum(y[:-1])

        # L(n): x_n * (-1-2*h2)
        f5 = (-1.0 - 2.0 * h2) * y[-1]

        # C(i): -kappa*h2 * cos(x_i)
        f6 = -self.kappa * h2 * jnp.sum(jnp.cos(y))

        return f1 + f2 + f3 + f4 + f5 + f6

    def y0(self):
        # Initial values from SIF file: i*h for i=1..n
        h = 1.0 / (self.n + 1)
        return inexact_asarray(jnp.arange(1, self.n + 1)) * h

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None
