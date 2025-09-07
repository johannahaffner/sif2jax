from jaxtyping import Array, Float

from ._misra_base import MISRABase


class MISRA1D(MISRABase):
    """NIST Data fitting problem MISRA1D given as an inconsistent set of
    nonlinear equations.

    Fit: y = b1*b2*x*((1+b2*x)**(-1))
    """

    def model(
        self, b1: Float[Array, ""], b2: Float[Array, ""], x_data: Float[Array, "14"]
    ) -> Float[Array, "14"]:
        """Mathematical model: y = b1*b2*x / (1 + b2*x)"""
        return b1 * b2 * x_data / (1.0 + b2 * x_data)
