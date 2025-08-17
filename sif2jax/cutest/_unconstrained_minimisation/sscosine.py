import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SSCOSINE(AbstractUnconstrainedMinimisation):
    """Scaled version of COSINE function by Luksan et al.

    Another function with nontrivial groups and
    repetitious elements. This is a scaled version of COSINE by Luksan et al.

    Source: problem 50 in
    L. Luksan, C. Matonoha and J. Vlcek
    Modified CUTE problems for sparse unconstrained optimization
    Technical Report 1081
    Institute of Computer Science
    Academy of Science of the Czech Republic

    that is a scaled variant of
    N. Gould, private communication.

    SIF input: N. Gould, Nov 1997
               this version Nick Gould, June, 2013

    Classification: OUR2-AN-V-0

    TODO: Human review needed
    Attempts made: Similar structure to SCOSINE, likely has same scaling issues
    Suspected issues: Same fundamental SIF interpretation problems as SCOSINE
    Resources needed: Fix SCOSINE first, then apply same solution to SSCOSINE
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Number of variables
    scal: float = 6.0  # Scaling parameter (different from SCOSINE which uses 12.0)

    def __init__(self, n: int = 5000):
        self.n = n

    def objective(self, y, args):
        del args
        n_minus_1 = self.n - 1

        # Compute scale factors S(i)
        i = jnp.arange(self.n)
        s = jnp.exp(i / n_minus_1 * self.scal)

        # Elements: E(i) = S(i)^2 * x(i)^2 for i=1..n-1
        e_vals = s[:-1] ** 2 * y[:-1] ** 2

        # Groups: G(i) = cos((S(i+1) - 0.5) * x(i+1) + e_vals[i])
        # for i=1..n-1
        mult = s[1:] - 0.5
        g_vals = jnp.cos(mult * y[1:] + e_vals)

        return jnp.sum(g_vals)

    @property
    def y0(self):
        # Starting point: x(i) = 1 / S(i)
        i = jnp.arange(self.n)
        s = jnp.exp(i / (self.n - 1) * self.scal)
        return 1.0 / s

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        # Minimum value is -(n-1)
        return jnp.array(-(self.n - 1))
