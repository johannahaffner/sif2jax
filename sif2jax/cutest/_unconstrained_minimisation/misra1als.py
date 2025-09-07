from jax import numpy as jnp
from jaxtyping import Array, Float

from ._misra_ls_base import MISRALSBase


class MISRA1ALS(MISRALSBase):
    """NIST Data fitting problem MISRA1A formulated as least squares minimization.

    Fit: y = b1*(1-exp[-b2*x])
    """

    def model(
        self, b1: Float[Array, ""], b2: Float[Array, ""], x_data: Float[Array, "14"]
    ) -> Float[Array, "14"]:
        """Mathematical model: y = b1*(1-exp[-b2*x])"""
        return b1 * (1.0 - jnp.exp(-b2 * x_data))
