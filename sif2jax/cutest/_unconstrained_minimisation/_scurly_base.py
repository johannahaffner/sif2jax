import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class SCURLYBase(AbstractUnconstrainedMinimisation):
    """Base class for SCURLY functions.

    A scaled version of CURLY - a banded function with semi-bandwidth k and
    negative curvature near the starting point. Variables are exponentially
    scaled with ratio exp(12) ≈ 162,754 between smallest and largest scale factors.

    Source: Nick Gould, September 1997.

    Classification: OUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 10000  # Number of dimensions. Options listed in SIF file: 100, 1000
    k: int = 20  # Semi-bandwidth.
    scal: float = 12.0  # Scaling parameter

    def __init__(self, n: int = 10000, k: int = 20):
        self.n = n
        self.k = k

    def objective(self, y, args):
        del args

        # Compute scale factors S(i) = exp((i-1)/(n-1) * scal)
        i = inexact_asarray(jnp.arange(1, self.n + 1))
        s = jnp.exp((i - 1) / (self.n - 1) * self.scal)

        # Apply scaling to variables
        scaled_y = y * s

        # Efficient computation of banded matrix-vector product
        # Q[i] = sum(scaled_y[j] for j in range(i, min(i+k+1, n)))

        # Use convolution for efficient computation of sliding window sums
        # Pad scaled_y with zeros at the end to handle boundary
        padded_y = jnp.pad(scaled_y, (0, self.k), mode="constant", constant_values=0)

        # Create a kernel of ones with size k+1
        kernel = jnp.ones(self.k + 1)

        # Use 1D convolution to compute sliding window sums
        # mode='valid' gives us exactly n outputs
        q = jnp.convolve(padded_y, kernel, mode="valid")[: self.n]

        # Compute objective: Q * (Q * (Q^2 - 20) - 0.1)
        result = q * (q * (q**2 - 20) - 0.1)
        return jnp.sum(result)

    @property
    def y0(self):
        # Compute scale factors
        i = inexact_asarray(jnp.arange(1, self.n + 1))
        s = jnp.exp((i - 1) / (self.n - 1) * self.scal)

        # Starting point: x_i = 0.0001 * i * S(i) / (n+1)
        return 0.0001 * i * s / (self.n + 1)

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None

    @property
    def expected_objective_value(self):
        return None
