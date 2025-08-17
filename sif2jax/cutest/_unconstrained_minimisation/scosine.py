import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


class SCOSINE(AbstractUnconstrainedMinimisation):
    """Scaled version of COSINE function.

    Another function with nontrivial groups and
    repetitious elements. This is a scaled version of COSINE.

    Source: N. Gould, private communication.

    SIF input: N. Gould, Nov 1997

    Classification: OUR2-AN-V-0

    TODO: Human review needed
    Attempts made:
    1. Implemented based on SIF analysis:
       cos((SCALE(i+1) - 0.5) * X(i+1) + SCALE(i)² * X(i)²)
    2. Corrected sign to match positive expected value
    Suspected issues: Objective ~2.3x off (1883 vs 4387),
    Hessian completely wrong (5e21 difference)
    Resources needed: Deep analysis of element parameter usage and group scaling in SIF
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Number of variables
    scal: float = 12.0  # Scaling parameter

    def __init__(self, n: int = 5000):
        self.n = n

    def objective(self, y, args):
        del args
        n_minus_1 = self.n - 1

        # Compute scale factors
        i = jnp.arange(self.n)
        scale = jnp.exp(i / n_minus_1 * self.scal)

        # Elements: E(i) = scale(i)^2 * x(i)^2 for i=1..n-1
        e_vals = scale[:-1] ** 2 * y[:-1] ** 2

        # Groups: G(i) = cos((scale(i+1) - 0.5) * x(i+1) + e_vals[i])
        # for i=1..n-1
        mult = scale[1:] - 0.5
        g_vals = jnp.cos(mult * y[1:] + e_vals)

        return -jnp.sum(g_vals)

    @property
    def y0(self):
        # Starting point: x(i) = 1 / scale(i)
        i = jnp.arange(self.n)
        scale = jnp.exp(i / (self.n - 1) * self.scal)
        return 1.0 / scale

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        # Minimum value is (n-1)
        return jnp.array(self.n - 1)
