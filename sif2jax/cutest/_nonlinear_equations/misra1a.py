from jax import numpy as jnp
from jaxtyping import Array, Float

from ._misra_base import MISRABase


class MISRA1A(MISRABase):
    """NIST Data fitting problem MISRA1A given as an inconsistent set of
    nonlinear equations.

    Fit: y = b1*(1-exp[-b2*x])
    """

    def model(
        self, b1: Float[Array, ""], b2: Float[Array, ""], x_data: Float[Array, "14"]
    ) -> Float[Array, "14"]:
        """Mathematical model: y = b1*(1-exp[-b2*x])"""
        return b1 * (1.0 - jnp.exp(-b2 * x_data))
