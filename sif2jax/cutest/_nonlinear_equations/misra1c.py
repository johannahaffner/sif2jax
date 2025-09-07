from jaxtyping import Array, Float

from ._misra_base import MISRABase


class MISRA1C(MISRABase):
    """NIST Data fitting problem MISRA1C given as an inconsistent set of
    nonlinear equations.

    Fit: y = b1 * (1-(1+2*b2*x)**(-.5))
    """

    def model(
        self, b1: Float[Array, ""], b2: Float[Array, ""], x_data: Float[Array, "14"]
    ) -> Float[Array, "14"]:
        """Mathematical model: y = b1 * (1-(1+2*b2*x)**(-.5))"""
        return b1 * (1.0 - (1.0 + 2.0 * b2 * x_data) ** (-0.5))
