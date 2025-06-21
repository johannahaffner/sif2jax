import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

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
    mask: Array = eqx.field(init=False)  # Will be initialized in __init__

    def __init__(self, n: int = 10000, k: int = 20):
        def create_mask(n, k):
            row_indices = jnp.arange(n)[:, None]
            col_indices = jnp.arange(n)[None, :]

            # A cell (i,j) should be 1 if i ≤ j ≤ min(i+k, n-1)
            min_indices = jnp.minimum(row_indices + k, n - 1)
            mask = (col_indices >= row_indices) & (col_indices <= min_indices)
            return mask

        self.n = n
        self.k = k
        self.mask = create_mask(n, k)

    def objective(self, y, args):
        del args
        q = self.mask.astype(y.dtype) @ y
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
