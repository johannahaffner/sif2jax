import jax.numpy as jnp
from jax import Array

from ..._misc import inexact_asarray
from ..._problem import AbstractUnconstrainedMinimisation


class CYCLIC3LS(AbstractUnconstrainedMinimisation):
    """
    The cyclic cubic system whose root at zero has exponential multiplicity
    as a function of dimension. This is the least-squares version.

    Source:  problem 8.2 in
    Wenrui Hao, Andrew J. Sommese and Zhonggang Zeng,
    "An algorithm and software for computing multiplicity structures
     at zeros of nonlinear systems", Technical Report,
    Department of Applied & Computational Mathematics & Statistics,
    University of Notre Dame, Indiana, USA (2012)

    SIF input: Nick Gould, Jan 2012.
    Least-squares version of CYCLIC3.SIF, Jan 2020.

    classification SUR2-AN-V-0
    """

    n_param: int = 100000  # Dimension parameter
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    def n(self) -> int:
        """Number of variables is N+2"""
        return self.n_param + 2

    def objective(self, y: Array, args) -> Array:
        """Compute the least-squares objective for the cyclic cubic system"""
        n = self.n_param

        # Vectorized computation for i = 1 to N: E(i) = x(i)^3 - x(i+1)*x(i+2)
        # Compute cubic terms for first n elements
        cubic_terms = y[:n] ** 3

        # Compute product terms: y[i+1] * y[i+2] for i = 0 to n-1
        product_terms = y[1 : n + 1] * y[2 : n + 2]

        # First n residuals
        residuals_n = cubic_terms - product_terms

        # E(N+1) = x(N+1) - x(1)
        residual_n_plus_1 = y[n] - y[0]

        # E(N+2) = x(N+2) - x(2)
        residual_n_plus_2 = y[n + 1] - y[1]

        # Combine all residuals
        residuals = jnp.concatenate(
            [residuals_n, jnp.array([residual_n_plus_1, residual_n_plus_2])]
        )

        # Return sum of squares (no 0.5 factor for least squares problems in CUTEst)
        return jnp.sum(residuals**2)

    @property
    def y0(self) -> Array:
        """Initial point: all 1000.0 as specified in SIF"""
        return inexact_asarray(jnp.full(self.n, 1000.0))

    @property
    def args(self):
        return None

    @property
    def expected_result(self) -> None:
        """Solution at zero has exponential multiplicity"""
        return None

    @property
    def expected_objective_value(self) -> Array:
        """Expected optimal objective value: 0"""
        return inexact_asarray(0.0)
