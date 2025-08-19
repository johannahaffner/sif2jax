import jax.numpy as jnp
from jax import config

from .liswet_base import _AbstractLISWET


config.update("jax_enable_x64", True)


class LISWET1(_AbstractLISWET):
    """LISWET1 - Li and Swetits k-convex approximation problem.

    A k-convex approximation problem posed as a convex quadratic problem.

    minimize 1/2 sum_{i=1}^{n+k} (x_i - c_i)^2

    subject to:
    sum_{i=0}^k C(k,i) (-1)^{k-i} x_{j+i} >= 0  for j=1,...,n

    where c_i = g(t_i) + small perturbation, t_i = (i-1)/(n+k-1)
    Case 1: g(t) = sqrt(t)

    Source:
    W. Li and J. Swetits,
    "A Newton method for convex regression, data smoothing and
    quadratic programming with bounded constraints",
    SIAM J. Optimization 3 (3) pp 466-488, 1993.

    Classification: QLR2-AN-V-V
    Default dimensions: n=2000, k=2
    """

    def _g_function(self, t):
        """The g(t) function for LISWET1: g(t) = sqrt(t)."""
        return jnp.sqrt(t)
