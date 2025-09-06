from jaxtyping import Array, Float

from ._misra_ls_base import MISRALSBase


class MISRA1BLS(MISRALSBase):
    """NIST Data fitting problem MISRA1B formulated as least squares minimization.

    Fit: y = b1 * (1-(1+b2*x/2)**(-2))
    """

    def model(
        self, b1: Float[Array, ""], b2: Float[Array, ""], x_data: Float[Array, "14"]
    ) -> Float[Array, "14"]:
        """Mathematical model: y = b1 * (1-(1+b2*x/2)**(-2))"""
        return b1 * (1.0 - (1.0 + b2 * x_data / 2.0) ** (-2.0))
