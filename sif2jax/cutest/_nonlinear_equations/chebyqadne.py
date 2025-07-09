import jax
import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


class CHEBYQADNE(AbstractNonlinearEquations):
    """
    The Chebyquad problem using the exact formula for the
    shifted chebyshev polynomials.
    The Hessian is full.

    Source: problem 35 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#133 (p. 44).
    SIF input: Nick Gould, March 1990.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-AN-V-V
    """

    n: int = 100  # Number of variables (also M)
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def chebypol(self, x, i):
        """Evaluate the i-th shifted Chebyshev polynomial at x"""
        dif = 2.0 * x - 1.0
        # Use arccos to compute Chebyshev polynomial
        # Convert i to same dtype as x to avoid promotion issues
        i_float = jnp.asarray(i, dtype=x.dtype)
        acosx = i_float * jnp.arccos(dif)
        return jnp.cos(acosx)

    def residual(self, y: Array, args) -> Array:
        """Compute the residuals"""
        n = self.n
        m = n  # M = N in this problem

        # Create array of indices 1 to m as floats to avoid dtype issues
        indices = jnp.arange(1, m + 1, dtype=y.dtype)

        # Vectorized computation of Chebyshev polynomials for all i and all y[j]
        # First, vmap over y to compute chebypol for each y[j] and all i values
        cheby_vals_per_y = jax.vmap(
            lambda yj: jax.vmap(lambda i: self.chebypol(yj, i))(indices)
        )(y)

        # Sum over all y values and divide by n
        residuals = jnp.sum(cheby_vals_per_y, axis=0) / n

        # Subtract constants for even indices
        is_even = (indices % 2) == 0
        i_squared = indices * indices
        constants = jnp.where(is_even, -1.0 / (i_squared - 1.0), 0.0)
        residuals = residuals - constants

        return residuals

    def y0(self) -> Array:
        """Initial guess for the optimization problem."""
        # Starting point: j/(n+1) for j = 1, ..., n
        n = self.n
        return jnp.arange(1, n + 1, dtype=jnp.float64) / (n + 1)

    def args(self):
        """Additional arguments for the residual function."""
        return None

    def expected_result(self) -> Array | None:
        """Expected result of the optimization problem."""
        # No exact solution provided in SIF file
        return None

    def expected_objective_value(self) -> Array | None:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def bounds(self) -> tuple[Array, Array] | None:
        """Bounds on variables: all in [0, 1]"""
        n = self.n
        lower = jnp.zeros(n, dtype=jnp.float64)
        upper = jnp.ones(n, dtype=jnp.float64)
        return lower, upper
