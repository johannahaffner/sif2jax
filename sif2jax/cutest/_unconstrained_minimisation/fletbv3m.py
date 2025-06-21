import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class FLETBV3M(AbstractUnconstrainedMinimisation):
    """The FLETBV3M function.

    Variant of FLETCBV3, another boundary value problem, by Luksan et al

    Source: problem 30 in
    L. Luksan, C. Matonoha and J. Vlcek
    Modified CUTE problems for sparse unconstrained optimization
    Technical Report 1081
    Institute of Computer Science
    Academy of Science of the Czech Republic

    based on a scaled version of the first problem given by
    R. Fletcher,
    "An optimal positive definite update for sparse Hessian matrices"
    Numerical Analysis report NA/145, University of Dundee, 1992.

    SIF input: Nick Gould, June, 2013
    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Default dimension in SIF file
    kappa: float = 1.0  # Parameter used in the problem
    objscale: float = 1.0e8  # Scaling factor

    def objective(self, y, args):
        del args
        h = 1.0 / (self.n + 1)
        h2 = h * h
        p = 1.0 / self.objscale

        # Define each term based on the SIF file
        # G(0): p * 0.5 * (x_1)^2
        f1 = 0.5 * p * (y[0]) ** 2

        # G(i) for i=1...n-1: p * 0.5 * (x_i - x_{i+1})^2
        f2 = 0.5 * p * jnp.sum((y[:-1] - y[1:]) ** 2)

        # G(n): p * 0.5 * (x_n)^2
        f3 = 0.5 * p * (y[-1]) ** 2

        # C(i): p * cos(x_i) * (-kappa/h^2)
        f4 = p * (-self.kappa / h2) * jnp.sum(jnp.cos(y))

        # S(i): 100 * sin(0.01 * x_i) * p * (1+2/h^2) - fix sign like FLETCBV3
        f5 = p * (1.0 + 2.0 / h2) * jnp.sum(100.0 * jnp.sin(0.01 * y))

        return f1 + f2 + f3 + f4 + f5

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
