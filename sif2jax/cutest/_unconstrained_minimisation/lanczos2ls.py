import jax.numpy as jnp

from ._lanczos_base import _AbstractLanczos


# TODO: Human review needed to verify the implementation matches the problem definition
class LANCZOS2LS(_AbstractLanczos):
    """NIST Data fitting problem LANCZOS2.

    In LANCZOS2, the y values are provided directly in the SIF file.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Dependent variable values (y) - specific to LANCZOS2
    def _data(self):
        y = jnp.array(
            [
                2.51340,
                2.04433,
                1.66840,
                1.36642,
                1.12323,
                0.92689,
                0.76793,
                0.63888,
                0.53378,
                0.44794,
                0.37759,
                0.31974,
                0.27201,
                0.23250,
                0.19966,
                0.17227,
                0.14934,
                0.13007,
                0.11381,
                0.10004,
                0.08833,
                0.07834,
                0.06977,
                0.06239,
            ]
        )
        return y

    def expected_result(self):
        return None
