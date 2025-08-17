from ._scurly_base import SCURLYBase


class SCURLY10(SCURLYBase):
    """The SCURLY10 function.

    A scaled version of CURLY10 - a banded function with semi-bandwidth 10 and
    negative curvature near the starting point. Variables are exponentially
    scaled with ratio exp(12) ≈ 162,754 between smallest and largest scale factors.

    Source: Nick Gould.

    SIF input: Nick Gould, September 1997.
    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def __init__(self, n: int = 10000, k: int = 10):
        super().__init__(n=n, k=k)
