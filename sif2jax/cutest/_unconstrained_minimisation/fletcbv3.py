import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: this has not yet been compared against another interface to CUTEst
class FLETCBV3(AbstractUnconstrainedMinimisation):
    """The FLETCBV3 function.

    A boundary value problem from Fletcher (1992).

    Source: The first problem given by
    R. Fletcher, "An optimal positive definite update for sparse Hessian matrices"
    Numerical Analysis report NA/145, University of Dundee, 1992.

    Note J. Haffner --------------------------------------------------------------------
    The reference given appears to be incorrect, the PDF available under the title above
    does not include a problem description.

    This can be defined for different dimensions (original SIF allows 10, 100, 1000,
    5000, or 10000), with 5000 being the default in the SIF file.
    ------------------------------------------------------------------------------------

    SIF input: Nick Gould, Oct 1992.
    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 5000  # Default dimension in SIF file
    scale: float = 1e8  # Called OBJSCALE in the SIF file
    extra_term: int = 1  # Corresponds to the parameter kappa, which is 1 or 0

    def objective(self, y, args):
        p, kappa = args
        h = 1.0 / (self.n + 1)
        h2 = h * h

        # From AMPL: put p scaling back and try flipping the sign on linear term
        term1 = 0.5 * p * (y[0]) ** 2
        term2 = jnp.sum(0.5 * p * (y[:-1] - y[1:]) ** 2)
        term3 = 0.5 * p * (y[-1]) ** 2
        term4 = jnp.sum(p * (1.0 + 2.0 / h2) * y)  # Note: sign flipped
        term5 = jnp.sum(-kappa * p * jnp.cos(y) / h2)

        return term1 + term2 + term3 + term4 + term5

    def y0(self):
        n = self.n
        h = 1.0 / (self.n + 1)
        # Starting point according to SIF file: i*h for i=1..n
        return inexact_asarray(jnp.arange(1, n + 1)) * h

    def args(self):
        # p and kappa from SIF file
        p = 1.0 / self.scale
        kappa = float(self.extra_term)
        return jnp.array([p, kappa])

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None  # Takes different values for different problem configurations
