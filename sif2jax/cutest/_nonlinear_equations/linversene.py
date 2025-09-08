import jax.numpy as jnp
from jax import Array

from ..._problem import AbstractNonlinearEquations


# TODO: Human review needed
# Attempts made: 10+ attempts over multiple implementations
# Suspected issues:
#   1. Python loop in residual computation causes timeout for n=1000
#   2. Vectorization attempts still timeout during test execution
#   3. Constraint values have scaling discrepancies (factor of 2) in some elements
#   4. Complex SIF structure with groups O(J,J), O(J+1,J), O(J+2,J) pattern
# Resources needed:
#   1. More efficient JAX vectorization strategy for the constraint pattern
#   2. Investigation of why pre-computed sin/cos values still lead to timeout
#   3. Verification of ROOTP5 scaling application in element computations


class LINVERSENE(AbstractNonlinearEquations):
    """
    The problem is to find the positive definite lower bidiagonal
    matrix L such that the matrix L(inv)L(inv-transp) best approximates,
    in the Frobenius norm, a given symmetric target matrix T.
    More precisely, one is interested in the positive definite lower
    bidiagonal L such that

         || L T L(transp) - I ||     is minimum.
                                F

    The positive definite character of L is imposed by requiring
    that all its diagonal entries to be at least equal to EPSILON,
    a strictly positive real number.

    Many variants of the problem can be obtained by varying the target
    matrix T and the scalar EPSILON. In the present problem,
    a) T is chosen to be pentadiagonal with T(i,j) = sin(i)cos(j) (j .leq. i)
    b) EPSILON = 1.D-8

    Source:
    Ph. Toint, private communication, 1991.

    SIF input: Ph. Toint, March 1991.
    Bound-constrained nonlinear equations version: Nick Gould, June 2019.

    classification NOR2-AN-V-V
    """

    n: int = 1000  # Dimension of the matrix
    epsilon: float = 1.0e-8
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def num_residuals(self) -> int:
        """Number of residuals."""
        # The number of groups O(i,j) equals n + (n-1) + (n-2) = 3n - 3
        return 3 * self.n - 3

    def _get_target_matrix_element(self, i: int, j: int):
        """Compute T(i,j) = sin(i)cos(j) for pentadiagonal matrix."""
        if abs(i - j) > 2:
            return 0.0
        return jnp.sin(float(i)) * jnp.cos(float(j))

    def residual(self, y: Array, args) -> Array:
        """Compute the residual vector following the exact SIF structure.

        The SIF defines groups as:
        DO J=1,N-2: XE O(J,J), ZE O(J+1,J) ROOTP5, ZE O(J+2,J) ROOTP5
        Plus: XE O(N-1,N-1), ZE O(N,N-1) ROOTP5, XE O(N,N)
        """
        n = self.n
        rootp5 = jnp.sqrt(0.5)

        # Extract A and B from the variable vector
        a = jnp.zeros(n, dtype=y.dtype)
        a = a.at[:-1].set(y[0 : 2 * (n - 1) : 2])  # A(1) to A(N-1)
        a = a.at[-1].set(y[-1])  # A(N) at last position
        b = y[1 : 2 * (n - 1) : 2]  # B(1) to B(N-1)

        # Pre-compute T matrix: T[i-1,j-1] = sin(i) * cos(j) for i,j = 1..n
        indices = jnp.arange(1, n + 1)
        sin_vals = jnp.sin(indices.astype(y.dtype))
        cos_vals = jnp.cos(indices.astype(y.dtype))

        # Build constraint list systematically following SIF structure
        constraints = []

        # Main SIF loop: DO J=1,N-2
        for j in range(1, n - 1):  # J = 1 to N-2 (1-based)
            j0 = j - 1  # Convert to 0-based

            # O(J,J): diagonal constraint (XE, unscaled, subtract 1)
            s_jj = a[j0] * a[j0] * sin_vals[j - 1] * cos_vals[j - 1]
            if j > 1:
                u_jj = a[j0] * b[j0 - 1] * sin_vals[j - 1] * cos_vals[j - 2]
                v_jj = u_jj  # Same element type
                w_jj = b[j0 - 1] * b[j0 - 1] * sin_vals[j - 2] * cos_vals[j - 2]
                constraints.append(s_jj + u_jj + v_jj + w_jj - 1.0)
            else:
                constraints.append(s_jj - 1.0)

            # O(J+1,J): off-diagonal constraint (ZE, scaled by ROOTP5)
            j1 = j + 1
            if j1 <= n:
                j1_0 = j1 - 1
                s_j1j = a[j1_0] * a[j0] * sin_vals[j1 - 1] * cos_vals[j - 1]
                if j1 > 1:
                    v_j1j = b[j1_0 - 1] * a[j0] * sin_vals[j1 - 2] * cos_vals[j - 1]
                    constraints.append(rootp5 * (s_j1j + v_j1j))
                else:
                    constraints.append(rootp5 * s_j1j)

            # O(J+2,J): off-diagonal constraint (ZE, scaled by ROOTP5)
            j2 = j + 2
            if j2 <= n:
                j2_0 = j2 - 1
                s_j2j = a[j2_0] * a[j0] * sin_vals[j2 - 1] * cos_vals[j - 1]
                if j2 > 1:
                    v_j2j = b[j2_0 - 1] * a[j0] * sin_vals[j2 - 2] * cos_vals[j - 1]
                    constraints.append(rootp5 * (s_j2j + v_j2j))
                else:
                    constraints.append(rootp5 * s_j2j)

        # Final constraints from SIF
        # O(N-1,N-1): XE (unscaled, subtract 1)
        s_n1n1 = a[n - 2] * a[n - 2] * sin_vals[n - 2] * cos_vals[n - 2]
        u_n1n1 = a[n - 2] * b[n - 3] * sin_vals[n - 2] * cos_vals[n - 3]
        v_n1n1 = u_n1n1
        w_n1n1 = b[n - 3] * b[n - 3] * sin_vals[n - 3] * cos_vals[n - 3]
        constraints.append(s_n1n1 + u_n1n1 + v_n1n1 + w_n1n1 - 1.0)

        # O(N,N-1): ZE (scaled by ROOTP5)
        s_nn1 = a[n - 1] * a[n - 2] * sin_vals[n - 1] * cos_vals[n - 2]
        v_nn1 = b[n - 2] * a[n - 2] * sin_vals[n - 2] * cos_vals[n - 2]
        constraints.append(rootp5 * (s_nn1 + v_nn1))

        # O(N,N): XE (unscaled, subtract 1)
        s_nn = a[n - 1] * a[n - 1] * sin_vals[n - 1] * cos_vals[n - 1]
        u_nn = a[n - 1] * b[n - 2] * sin_vals[n - 1] * cos_vals[n - 2]
        v_nn = u_nn
        w_nn = b[n - 2] * b[n - 2] * sin_vals[n - 2] * cos_vals[n - 2]
        constraints.append(s_nn + u_nn + v_nn + w_nn - 1.0)

        return jnp.array(constraints)

    @property
    def y0(self) -> Array:
        num_vars = 2 * self.n - 1  # Variables are A(1)..A(N) and B(1)..B(N-1)
        return jnp.full(num_vars, -1.0, dtype=jnp.float64)

    @property
    def args(self):
        """Additional arguments for the residual function."""
        return None

    @property
    def expected_result(self) -> Array:
        """Expected result of the optimization problem."""
        # Not explicitly given
        num_vars = 2 * self.n - 1
        return jnp.zeros(num_vars, dtype=jnp.float64)

    @property
    def expected_objective_value(self) -> Array:
        """Expected value of the objective at the solution."""
        # For nonlinear equations with pycutest formulation, this is always zero
        return jnp.array(0.0)

    def constraint(self, y):
        """Returns the residuals as equality constraints."""
        return self.residual(y, self.args), None

    @property
    def bounds(self) -> tuple[Array, Array] | None:
        """Bounds for variables - A(i) >= epsilon, B(i) free."""
        num_vars = 2 * self.n - 1
        lower = jnp.full(num_vars, -jnp.inf, dtype=jnp.float64)
        upper = jnp.full(num_vars, jnp.inf, dtype=jnp.float64)

        # Set lower bounds for A(i) variables using vectorized operations
        # A(i) variables are at positions 0, 2, 4, ..., 2*(n-1)
        a_indices = jnp.concatenate(
            [jnp.arange(self.n - 1) * 2, jnp.array([2 * (self.n - 1)])]
        )
        lower = lower.at[a_indices].set(self.epsilon)

        return lower, upper
