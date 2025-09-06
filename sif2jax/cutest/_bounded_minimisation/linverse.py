# TODO: Human review needed
# Attempts made:
# 1. Initial implementation with Python loops
# 2. First vectorization attempt - incorrect structure understanding
# 3. Analyzed SIF GROUP USES section - still incorrect
# 4. Multiple vectorization attempts with jnp.where
# 5. Corrected variable ordering (interleaved A,B variables)
# 6. Fixed bounds for interleaved variables
# 7. Simplified vectorized approach focusing on key groups
# Suspected issues:
# - Complex SIF structure with multiple element types (S, U, V, W)
# - Matrix structure L T L^T computation not correctly understood
# - GROUP USES section maps elements to T matrix coefficients in non-obvious way
# Resources needed:
# - Detailed analysis of pentadiagonal structure
# - Understanding of how S,U,V,W elements map to L T L^T computation
# - Potentially need to implement full matrix construction approach

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..._problem import AbstractBoundedMinimisation


class LINVERSE(AbstractBoundedMinimisation):
    """LINVERSE problem.

    The problem is to find the positive definite lower bidiagonal
    matrix L such that the matrix L(inv)L(inv-transp) best approximates,
    in the Frobenius norm, a given symmetric target matrix T.
    More precisely, one is interested in the positive definite lower
    bidiagonal L such that

         || L T L(transp) - I ||_F     is minimum.

    The positive definite character of L is imposed by requiring
    that all its diagonal entries to be at least equal to EPSILON,
    a strictly positive real number.

    Many variants of the problem can be obtained by varying the target
    matrix T and the scalar EPSILON.  In the present problem,
    a) T is chosen to be pentadiagonal with T(i,j) = sin(i)cos(j) (j .leq. i)
    b) EPSILON = 1.D-8

    Source:
    Ph. Toint, private communication, 1991.

    SIF input: Ph. Toint, March 1991.

    classification SBR2-AN-V-0
    """

    # Default parameter
    N: int = 1000  # Dimension of the matrix

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        """Compute the objective function."""
        del args

        n = self.N

        # Extract variables: interleaved A(i), B(i) for i=1..n-1, then A(n)
        # Variables: A(1), B(1), A(2), B(2), ..., A(N-1), B(N-1), A(N)
        a = jnp.zeros(n, dtype=y.dtype)
        a = a.at[:-1].set(y[::2][: n - 1])  # A(1) to A(N-1)
        a = a.at[-1].set(y[-1])  # A(N) at last position
        b = y[1::2]  # B(1) to B(N-1)

        # Compute all sin and cos values (1-based indexing)
        idx = jnp.arange(1, n + 1, dtype=y.dtype)
        sin_vals = jnp.sin(idx)
        cos_vals = jnp.cos(idx)

        # Vectorized implementation based on SIF file structure
        # The objective is || L T L^T - I ||_F^2 where L is lower bidiagonal

        # 1. All diagonal groups O(j,j) contribute (a[j]^2 * T[j,j] - 1)^2
        t_diag = sin_vals * cos_vals  # T[j,j] = sin(j+1) * cos(j+1)
        residuals_diag = a * a * t_diag - 1.0
        obj = jnp.sum(residuals_diag * residuals_diag)

        # 2. First subdiagonal O(j+1,j) for j=0 to n-2 with scale 0.5
        # S(j+1,j) = a[j+1] * a[j] * T[j+1,j]
        # V(j+1,j) = b[j] * a[j] * T[j,j]
        s_sub1 = a[1:] * a[:-1] * sin_vals[1:] * cos_vals[:-1]  # T[j+1,j]
        v_sub1 = b * a[:-1] * sin_vals[:-1] * cos_vals[:-1]  # T[j,j]
        residuals_sub1 = 0.5 * (s_sub1 + v_sub1)
        obj += jnp.sum(residuals_sub1 * residuals_sub1)

        # 3. Second subdiagonal O(j+2,j) for j=0 to n-3 with scale 0.5
        # Only add this contribution if n > 2
        second_sub_contrib = jnp.where(
            n > 2,
            jnp.sum(
                (
                    0.5
                    * (
                        a[2:] * a[:-2] * sin_vals[2:] * cos_vals[:-2]  # S(j+2,j)
                        + b[1:] * a[:-2] * sin_vals[1:-1] * cos_vals[:-2]  # V(j+2,j)
                    )
                )
                ** 2
            ),
            0.0,
        )
        obj += second_sub_contrib

        return obj

    @property
    def y0(self):
        """Initial guess for variables."""
        # All variables start at -1.0
        return jnp.full(2 * self.N - 1, -1.0)

    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file comments:
        # SOLTN(10)   = 6.00000000
        # SOLTN(100)  = 68.0000000
        # SOLTN(500)  = 340.000000
        # SOLTN(1000) = ???
        return None

    @property
    def n(self):
        """Number of variables."""
        return 2 * self.N - 1

    @property
    def bounds(self) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
        """Lower and upper bounds for variables.

        The diagonal entries a[i] have lower bound EPSILON = 1e-8.
        The off-diagonal entries b[i] have no bounds.
        Variables are interleaved: A(1), B(1), A(2), B(2), ..., A(N-1), B(N-1), A(N)
        """
        epsilon = 1.0e-8
        n = self.N
        lower = jnp.full(2 * n - 1, -jnp.inf)

        # Set bounds for A variables (at even indices and last position)
        lower = lower.at[::2].set(epsilon)  # A(1) to A(N-1) at positions 0, 2, 4, ...
        lower = lower.at[-1].set(epsilon)  # A(N) at last position

        upper = jnp.full(2 * n - 1, jnp.inf)
        return lower, upper
