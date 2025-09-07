from jaxtyping import Array, Float

from ._misra_ls_base import MISRALSBase


class MISRA1DLS(MISRALSBase):
    """NIST Data fitting problem MISRA1D formulated as least squares minimization.

    Fit: y = b1*b2*x*((1+b2*x)**(-1))
    """

    def model(
        self, b1: Float[Array, ""], b2: Float[Array, ""], x_data: Float[Array, "14"]
    ) -> Float[Array, "14"]:
        """Mathematical model: y = b1*b2*x / (1 + b2*x)"""
        return b1 * b2 * x_data / (1.0 + b2 * x_data)
