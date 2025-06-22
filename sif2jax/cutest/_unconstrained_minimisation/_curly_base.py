import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


# TODO: this should still be compared against another CUTEst interface
class CURLYBase(AbstractUnconstrainedMinimisation):
    """Base class for CURLY functions.

    A banded function with semi-bandwidth k and
    negative curvature near the starting point.

    Note J. Haffner --------------------------------------------------------------------
    The value q is created by the matrix-vector product of the mask and y. The mask has
    the form:

    [***  ]
    [ *** ]
    [  ***]
    [   **]
    [    *]

    And q = M @ y.
    ------------------------------------------------------------------------------------

    Source: Nick Gould, September 1997.

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 10000  # Number of dimensions. Options listed in SIF file: 100, 1000
    k: int = 20  # Semi-bandwidth.

    def __init__(self, n: int = 10000, k: int = 20):
        self.n = n
        self.k = k

    def objective(self, y, args):
        del args

        # Efficient computation of banded matrix-vector product
        # q[i] = sum(y[j] for j in range(i, min(i+k+1, n)))

        # Use convolution for efficient computation of sliding window sums
        # Pad y with zeros at the end to handle boundary
        padded_y = jnp.pad(y, (0, self.k), mode="constant", constant_values=0)

        # Create a kernel of ones with size k+1
        kernel = jnp.ones(self.k + 1)

        # Use 1D convolution to compute sliding window sums
        # mode='valid' gives us exactly n outputs
        q = jnp.convolve(padded_y, kernel, mode="valid")[: self.n]

        result = q * (q * (q**2 - 20) - 0.1)
        return jnp.sum(result)

    def y0(self):
        # Use float to ensure proper dtype promotion
        i = inexact_asarray(jnp.arange(1, self.n + 1))
        return 0.0001 * i / (self.n + 1)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None
