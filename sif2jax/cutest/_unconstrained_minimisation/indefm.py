import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class INDEFM(AbstractUnconstrainedMinimisation):
    """
    INDEFM problem.

    Variant of INDEF, a nonconvex problem which has an indefinite Hessian
    at the starting point, by Luksan et al

    Source: problem 37 in
    L. Luksan, C. Matonoha and J. Vlcek
    Modified CUTE problems for sparse unconstrained optimization
    Technical Report 1081
    Institute of Computer Science
    Academy of Science of the Czech Republic

    based on the original problem by N. Gould

    SIF input: Nick Gould, June, 2013

    classification OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    n: int = 100000
    alpha: float = 0.5

    def objective(self, y, args):
        del args
        n = self.n
        alpha = self.alpha

        # SIN groups: 100.0 * sin(0.01 * x[i]) for i = 0 to n-1
        sin_terms = 100.0 * jnp.sin(0.01 * y)
        sin_sum = jnp.sum(sin_terms)

        # COS groups: alpha * cos(2*x[i] - x[0] - x[n-1]) for i = 1 to n-2
        if n > 2:
            cos_args = 2.0 * y[1 : n - 1] - y[0] - y[n - 1]
            cos_terms = alpha * jnp.cos(cos_args)
            cos_sum = jnp.sum(cos_terms)
        else:
            cos_sum = 0.0

        return sin_sum + cos_sum

    def y0(self):
        # Starting point: X(I) = I/(N+1) for INDEF1
        # INDEF2 has all variables at 1000.0
        n = self.n
        return inexact_asarray(jnp.arange(1.0, n + 1.0)) / inexact_asarray(n + 1.0)

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in SIF file
        return None

    def expected_objective_value(self):
        # Solution value not provided in SIF file
        return None

    def num_variables(self):
        return self.n
