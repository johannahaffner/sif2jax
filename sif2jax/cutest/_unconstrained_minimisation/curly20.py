from ._curly_base import CURLYBase


# TODO: this should still be compared against another CUTEst interface
class CURLY20(CURLYBase):
    """The CURLY 20 function.

    A banded function with semi-bandwidth 20 and
    negative curvature near the starting point.

    Source: Nick Gould, September 1997.

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n: int = 10000, k: int = 20):
        super().__init__(n=n, k=k)
